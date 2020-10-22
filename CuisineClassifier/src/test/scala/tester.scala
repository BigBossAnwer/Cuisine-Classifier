import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.log4j.LogManager
import org.apache.spark.ml.feature.{CountVectorizer, IDF, IndexToString, StringIndexer}
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import scala.collection.mutable.ListBuffer
//import org.apache.hadoop.conf.Configuration
//import org.apache.hadoop.fs._
//import java.io.File

object tester {
  /*  def merge(srcPath: String, dstPath: String): Unit = {
      val hadoopConfig = new Configuration()
      val hdfs = FileSystem.get(hadoopConfig)
      FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath),
        true, hadoopConfig, null)
    }*/

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: CuisinePipeline Input.json ResultsDir")
      System.exit(0)
    }

    val spark = SparkSession.builder()
      //.master("local[*]")
      .appName("CuisinePipelineTest")
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

    // -------------------
    // End: Preparing data

    // Start: ML pipeline
    // ------------------

    // Using linear SVM as the base classifier
    val baseSVM = new LinearSVC()
    //Debug:
    //.setMaxIter(1)
    //.setRegParam(1)
    //.setTol(10)

    // Using One vs. All technique on top of linear SVM to handle multiple target classes
    val ovrClassifier = new OneVsRest()
      .setClassifier(baseSVM)

    val pipeline = new Pipeline()
      .setStages(Array(ovrClassifier))

    // Using a grid search for parameter tuning
    val paramGrid = new ParamGridBuilder()
      .addGrid(baseSVM.maxIter, Array(1, 2))
      .addGrid(baseSVM.regParam, Array(1.0))
      .addGrid(baseSVM.tol, Array(10.0))
      .build()

    // Using 5 fold cross validation for model evaluation
    /*    val cv = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(new MulticlassClassificationEvaluator())
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(2)
          .setParallelism(2)*/

    val tv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)

    // Run through pipeline & get best model by cross validation
    LogManager.getLogger("myLog").warn("Fitting...")
    val cvModel = tv.fit(scaledData)

    // Debug:
    //val cvModel = ovrClassifier.fit(scaledData)

    //----------------
    // End ML Pipeline

    // Start: Report metrics
    //----------------------
    LogManager.getLogger("myLog").warn("Collecting results...")
    val predictionAndLabels = cvModel.transform(scaledData)
      .select("prediction", "label")
      .map(r => (r.getDouble(0), r.getDouble(1))).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val labels = metrics.labels


    //val optParams = Seq(cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2)._1.toString())
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

    def metricDecodeToCSV(decoder: IndexToString,
                          df: DataFrame,
                          metric: String): Unit = {
      decoder.transform(df)
        .select("label", "cuisine label", metric)
        .orderBy(desc(metric))
        .write.format("csv")
        .option("header", value = true)
        .save(args(1) + "/" + metric)
    }

    /*    FileUtil.fullyDelete(new File(args(1) + "/optParams"))
        FileUtil.fullyDelete(new File(args(1) + "/confusionMatrix"))
        FileUtil.fullyDelete(new File(args(1) + "/precision"))
        FileUtil.fullyDelete(new File(args(1) + "/recall"))
        FileUtil.fullyDelete(new File(args(1) + "/fpr"))
        FileUtil.fullyDelete(new File(args(1) + "/f-measure"))
        FileUtil.fullyDelete(new File(args(1) + "/weightedMeasures"))*/

    // Store results in results dir
    //sc.parallelize(optParams).coalesce(1, shuffle = true).saveAsTextFile(args(1) + "/optParams")
    sc.parallelize(confusionMatrix).saveAsTextFile(args(1) + "/confusionMatrix")
    metricDecodeToCSV(decoder, precisionDf, "precision")
    metricDecodeToCSV(decoder, recallDf, "recall")
    metricDecodeToCSV(decoder, fprDf, "fpr")
    metricDecodeToCSV(decoder, fMeasureDf, "f-measure")
    sc.parallelize(weightedMeasures).saveAsTextFile(args(1) + "/weightedMeasures")

    // Merge part files for ease of analysis
    /*FileUtil.fullyDelete(new File(args(1) + "/optParams.txt"))
    FileUtil.fullyDelete(new File(args(1) + "/confusionMatrix.csv"))
    FileUtil.fullyDelete(new File(args(1) + "/precision.csv"))
    FileUtil.fullyDelete(new File(args(1) + "/recall.csv"))
    FileUtil.fullyDelete(new File(args(1) + "/fpr.csv"))
    FileUtil.fullyDelete(new File(args(1) + "/f-measure.csv"))
    FileUtil.fullyDelete(new File(args(1) + "/weightedMeasures.csv"))*/

    /*    merge(args(1) + "/optParams", args(1) + "/optParams.txt")
        merge(args(1) + "/confusionMatrix", args(1) + "/confusionMatrix.csv")
        merge(args(1) + "/precision", args(1) + "/precision.csv")
        merge(args(1) + "/recall", args(1) + "/recall.csv")
        merge(args(1) + "/fpr", args(1) + "/fpr.csv")
        merge(args(1) + "/f-measure", args(1) + "/f-measure.csv")
        merge(args(1) + "/weightedMeasures", args(1) + "/weightedMeasures.csv")*/

    LogManager.getLogger("myLog").warn("Finished successfully")
    spark.stop()
  }
}
