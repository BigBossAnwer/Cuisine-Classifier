import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._

object SquashParts {
  def merge(srcPath: String, dstPath: String): Unit = {
    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)
    FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath),
      true, hadoopConfig, null)
  }

  def main(args: Array[String]): Unit = {
    merge(args(0) + "/optParams", args(0) + "/optParams.txt")
    merge(args(0) + "/confusionMatrix", args(0) + "/confusionMatrix.csv")
    merge(args(0) + "/precision", args(0) + "/precision.csv")
    merge(args(0) + "/recall", args(0) + "/recall.csv")
    merge(args(0) + "/fpr", args(0) + "/fpr.csv")
    merge(args(0) + "/f-measure", args(0) + "/f-measure.csv")
    merge(args(0) + "/weightedMeasures", args(0) + "/weightedMeasures.csv")
  }
}
