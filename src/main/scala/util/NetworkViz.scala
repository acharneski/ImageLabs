/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package util

import java.util.UUID

import com.simiacryptus.mindseye.network.graph.{DAGNetwork, DAGNode, InnerNode}
import guru.nidi.graphviz.attribute.RankDir
import guru.nidi.graphviz.model.{Link, LinkSource, MutableNode}

import scala.collection.JavaConverters._

object NetworkViz {

  def getNodes(node: DAGNode): List[DAGNode] = node match {
    case network: DAGNetwork ⇒ network.getNodes.asScala.toList //.flatMap(getNodes(_))
    case _ ⇒ List(node)
  }

  def toGraph(network: DAGNetwork) = {
    val nodes: List[DAGNode] = network.getNodes.asScala.toList
    val graphNodes: Map[UUID, MutableNode] = nodes.map(node ⇒ {
      node.getId() → guru.nidi.graphviz.model.Factory.mutNode((node match {
        case n: InnerNode ⇒
          n.layer match {
            case _ ⇒ n.layer.getClass.getSimpleName
          }
        case _ ⇒ node.getClass.getSimpleName
      }) + "\n" + node.getId.toString)
    }).toMap
    val idMap: Map[UUID, List[UUID]] = nodes.flatMap((to: DAGNode) ⇒ {
      val inputs: List[DAGNode] = to.getInputs.toList
      inputs.map((from: DAGNode) ⇒ {
        if (null == from) throw new AssertionError();
        if (null == to) throw new AssertionError();
        from.getId → to.getId
      })
    }).groupBy(_._1).mapValues(_.map(_._2))
    nodes.foreach((to: DAGNode) ⇒ {
      graphNodes(to.getId).addLink(idMap.getOrElse(to.getId, List.empty).map(from ⇒ {
        Link.to(graphNodes(from))
      }): _*)
    })
    val nodeArray = graphNodes.values.map(_.asInstanceOf[LinkSource]).toArray
    guru.nidi.graphviz.model.Factory.graph().`with`(nodeArray: _*)
      .generalAttr.`with`(RankDir.TOP_TO_BOTTOM).directed()
  }

}
