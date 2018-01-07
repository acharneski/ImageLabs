/*
 * Copyright (c) 2018 by Andrew Charneski.
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

///*
// * Copyright (c) 2017 by Andrew Charneski.
// *
// * The author licenses this file to you under the
// * Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance
// * with the License.  You may obtain a copy
// * of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package interactive.superres
//
//import util.Report
//
///**
//  * Created by Andrew Charneski on 7/22/2017.
//  */
//object ModelScript extends Report {
//
//  def main(args: Array[String]): Unit = {
//
//    report((server, out) ⇒ args match {
//      case Array(source) ⇒
//        new DownsamplingModel(source, server, out).run(false)
//        new DiscriminatorModel(source, server, out).run(false)
//        new UpsamplingOptimizer(source, server, out).run(false)
//        new UpsamplingModel(source, server, out).run(false)
//      case _ ⇒
//        new DownsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
//        new DiscriminatorModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
//        new UpsamplingOptimizer("E:\\testImages\\256_ObjectCategories", server, out).run(false)
//        new UpsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
//    })
//
//  }
//
//}
