import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.{CountVectorizer, IDF, IndexToString, StringIndexer}
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object CuisinePipeline {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: CuisinePipeline Input.json ResultsDir")
      System.exit(0)
    }

    val spark = SparkSession.builder()
      //.master("local[*]")
      .appName("CuisinePipeline")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    // Start: Preparing data
    //----------------------

    val rawDs = spark.createDataset(sc.wholeTextFiles(args(0)).values)
    val rawDf = spark.read.json(rawDs)

    // Clean ingredient features of Case Impurity
    val cleaned = rawDf
      .withColumn("ingredient", explode($"ingredients"))
      .withColumn("lowerIngredient", lower($"ingredient"))
      .groupBy("cuisine", "id")
      .agg(collect_list("lowerIngredient").alias("ingredients"))

    // Generate numerical Labels for target classes
    val labeler = new StringIndexer()
      .setInputCol("cuisine")
      .setOutputCol("label")
      .fit(cleaned)

    val labeledData = labeler.transform(cleaned)

    // Generate feature count vectors on ingredients for TF-IDF featurizing
    val countVectorizer = new CountVectorizer()
      .setInputCol("ingredients")
      .setOutputCol("rawFeatures")
      .setVocabSize(6703)
      .fit(labeledData)

    val featurizedData = countVectorizer.transform(labeledData)

    // Scale raw features using TF-IDF
    val idfer = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(featurizedData)

    val scaledData = idfer.transform(featurizedData)
      .select("label", "features")
      .cache()

    //--------------------
    // End: Preparing data

    // Start: ML pipeline
    //-------------------

    // Using linear SVM as the base classifier
    val baseSVM = new LinearSVC()

    // Using One vs. All technique on top of linear SVM to handle multiple target classes
    val ovrClassifier = new OneVsRest()
      .setClassifier(baseSVM)

    val pipeline = new Pipeline()
      .setStages(Array(ovrClassifier))

    // Using a grid search for parameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(baseSVM.maxIter, Array(20))
      .addGrid(baseSVM.regParam, Array(1, 0, .1, .01, .001))
      .addGrid(baseSVM.tol, Array(1E-3, 1E-4, 1E-5, 1E-6))
      .build()

    // Randomized one-shot validation split using 4:1 training ratio
    val tv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(.8)
      .setParallelism(3)

    // Run through pipeline & get best model by randomized one-shot validation
    LogManager.getLogger("myLog").warn("Fitting...")
    val tvModel = tv.fit(scaledData)

    //-----------------
    // End: ML Pipeline

    // Start: Report metrics
    //----------------------

    LogManager.getLogger("myLog").warn("Collecting results...")
    val predictionAndLabels = tvModel.transform(scaledData)
      .select("prediction", "label")
      .map(r => (r.getDouble(0), r.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val labels = metrics.labels

    val optParams = Seq(tvModel.getEstimatorParamMaps.zip(tvModel.validationMetrics).maxBy(_._2)._1.toString())
    var confusionMatrix = new ListBuffer[String]()
    metrics.confusionMatrix.rowIter.toSeq.foreach { l =>
      confusionMatrix += l.toArray.mkString(",")
    }

    var precision = new ListBuffer[(Double, Double)]()
    labels.foreach { l =>
      precision += ((l, metrics.precision(l)))
    }
    val precisionDf = precision.toDF("label", "precision")

    var recall = new ListBuffer[(Double, Double)]()
    labels.foreach { l =>
      recall += ((l, metrics.recall(l)))
    }
    val recallDf = recall.toDF("label", "recall")

    var fpr = new ListBuffer[(Double, Double)]()
    labels.foreach { l =>
      fpr += ((l, metrics.falsePositiveRate(l)))
    }
    val fprDf = fpr.toDF("label", "fpr")

    var fMeasure = new ListBuffer[(Double, Double)]()
    labels.foreach { l =>
      fMeasure += ((l, metrics.fMeasure(l)))
    }
    val fMeasureDf = fMeasure.toDF("label", "f-measure")

    var weightedMeasures = Seq("accuracy, weighted precision, weighted recall, weighted f-measure, weighted fpr")
    weightedMeasures ++= Seq(f"${metrics.accuracy}, ${metrics.weightedPrecision}, ${metrics.weightedRecall}, " +
      f"${metrics.weightedFMeasure}, ${metrics.weightedFalsePositiveRate}")

    val decoder = new IndexToString()
      .setInputCol("label")
      .setOutputCol("cuisine label")
      .setLabels(labeler.labels)

    def metricDecodeToCSV(decoder: IndexToString, df: DataFrame, metric: String): Unit = {
      decoder.transform(df)
        .select("label", "cuisine label", metric)
        .orderBy(desc(metric))
        .write.format("csv")
        .save(args(1) + "/" + metric)
    }

    // Store results in results dir
    sc.parallelize(optParams).saveAsTextFile(args(1) + "/optParams")
    sc.parallelize(confusionMatrix).saveAsTextFile(args(1) + "/confusionMatrix")
    metricDecodeToCSV(decoder, precisionDf, "precision")
    metricDecodeToCSV(decoder, recallDf, "recall")
    metricDecodeToCSV(decoder, fprDf, "fpr")
    metricDecodeToCSV(decoder, fMeasureDf, "f-measure")
    sc.parallelize(weightedMeasures).saveAsTextFile(args(1) + "/weightedMeasures")

    LogManager.getLogger("myLog").warn("Finished successfully")
    spark.stop()
  }
}
