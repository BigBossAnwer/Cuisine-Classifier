name := "CuisineClassifier"

version := "1.0-Prod"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.2"

val hadoopVersion = "2.7.3"

libraryDependencies ++= Seq(
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.hadoop" % "hadoop-hdfs" % hadoopVersion
)