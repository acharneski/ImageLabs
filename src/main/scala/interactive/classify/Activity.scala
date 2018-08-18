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
// * Copyright (c) 2018 by Andrew Charneski.
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
//package interactive.classify
//
//import java.awt.AWTEvent
//import java.awt.event.AWTEventListener
//
//import _root_.util._
//
//object Activity extends Report {
//
//  def main(args: Array[String]): Unit = {
//    new Activity().run()
//    Thread.sleep(60*1000*5)
//  }
//
//}
//
//
//
//class Activity() {
//  def run(): Unit = {
//    import java.awt.Toolkit
//    Toolkit.getDefaultToolkit.addAWTEventListener(new AWTEventListener() {
//      override def eventDispatched(event: AWTEvent): Unit = {
//        println(event.toString)
//      }
//    },
//      //-1)
//      AWTEvent.COMPONENT_EVENT_MASK |
//        AWTEvent.CONTAINER_EVENT_MASK |
//        AWTEvent.FOCUS_EVENT_MASK |
//        AWTEvent.KEY_EVENT_MASK |
//        AWTEvent.MOUSE_EVENT_MASK |
//        AWTEvent.MOUSE_MOTION_EVENT_MASK |
//        AWTEvent.WINDOW_EVENT_MASK |
//        AWTEvent.ACTION_EVENT_MASK |
//        AWTEvent.ADJUSTMENT_EVENT_MASK |
//        AWTEvent.ITEM_EVENT_MASK |
//        AWTEvent.TEXT_EVENT_MASK |
//        AWTEvent.INPUT_METHOD_EVENT_MASK |
//        AWTEvent.PAINT_EVENT_MASK |
//        AWTEvent.INVOCATION_EVENT_MASK |
//        AWTEvent.HIERARCHY_EVENT_MASK |
//        AWTEvent.HIERARCHY_BOUNDS_EVENT_MASK |
//        AWTEvent.MOUSE_WHEEL_EVENT_MASK |
//        AWTEvent.WINDOW_STATE_EVENT_MASK |
//        AWTEvent.WINDOW_FOCUS_EVENT_MASK)
//  }
//}
//
