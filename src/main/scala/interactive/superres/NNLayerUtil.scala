package interactive.superres

import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper
import com.simiacryptus.util.MonitoredObject

/**
  * Created by Andrew Charneski on 7/20/2017.
  */
object NNLayerUtil {
  implicit def cast(inner:NNLayer) = new NNLayerUtil(inner)
}

case class NNLayerUtil(inner:NNLayer) {
  def withMonitor = new MonitoringWrapper(inner)
  def addTo(monitor: MonitoredObject) = withMonitor.addTo(monitor)
}