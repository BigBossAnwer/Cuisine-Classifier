# Cuisine Classification Pipeline

An accurate, low-latency Spark Streaming Dataset ML cuisine classifier pipeline for a recipe dataset provided by [Yummly.com](https://www.yummly.com/)

For a full discussion and analysis, see the [online report](https://htmlpreview.github.io/?https://github.com/BigBossAnwer/Cuisine-Classifier/blob/main/Report.html). Alternatively, download the [report](Report.html) and view it in your browser

## Prerequisites

* If using prebuilt .jar:
	* Java
* If building from source:
  * Scala 2.11.8
  * Spark 2.4.2
  * Hadoop 2.7.3
  * sbt *(see [build.sbt](CuisineClassifier/build.sbt) for dependencies)*

## Usage

*Prebuilt .jar targets Scala 2.11.8, Spark 2.4.2, Hadoop 2.7.3, and assumes running on a distributed cloud instance. Uncomment line 20 in CuisinePipeline.scala & build for local usage*

*See [Yummly Dataset Schema](https://www.kaggle.com/c/whats-cooking/data) for assumed input.json data schema (alternatively see the included [train.json](CuisineClassifier/resources/train.json) sample)*

Sample .jar usage:

```bash
java -jar /pathToJar/cuisineclassifier_2.11-1.0-Prod Input.json /pathToOutputDir
```

Sample AWS EMR .jar usage:

```bash
spark-submit --deploy-mode cluster --class CuisinePipeline s3://pathToJar/cuisineclassifier_2.11-1.0-Prod s3://pathToInput/input.json s3://pathToOutputDir
```

Helper usage:

```bash
scala SquashParts /pathToOutputDir
```

Where */pathToOutputDir* contains the various output directories housing the part files produced by CuisinePipeline:
- */pathToOutputDir/optParams*
- */pathToOutputDir/confusionMatrix*
- etc.

## File Listing

```
CuisineClassifier/...
   	cuisineclassifier_2.11-1.0-Prod - Prebuilt Project Jar
	build.sbt - Project dependencies sbt file
	src/...
		CuisinePipeline.scala - Core classifier pipeline source code
		SquashParts.scala - Helper object to squash CuisinePipeline output part files into consolidated CSV and text files
		tester.scala - Debugging and alternative solutions testbed class
	resources/
	    train.json - Sample input file used for reported result # Source: https://www.kaggle.com/c/whats-cooking/data
	out/
		* - Sample result files 
```
