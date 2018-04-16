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

import boofcv.abst.feature.describe.ConfigBrief;
import boofcv.abst.feature.describe.DescribeRegionPoint;
import boofcv.abst.feature.detect.interest.ConfigGeneralDetector;
import boofcv.abst.feature.detect.interest.InterestPointDetector;
import boofcv.alg.feature.detect.interest.GeneralFeatureDetector;
import boofcv.alg.filter.derivative.GImageDerivativeOps;
import boofcv.factory.feature.describe.FactoryDescribeRegionPoint;
import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
import boofcv.factory.feature.detect.interest.FactoryDetectPoint;
import boofcv.factory.feature.detect.interest.FactoryInterestPoint;
import boofcv.struct.image.ImageGray;

/**
 * Created by Andrew Charneski on 4/16/2017.
 */
public class test {
  {
    // create a corner detector
    Class<? extends ImageGray> imageType = null;
    Class derivType = GImageDerivativeOps.getDerivativeType(imageType);
    GeneralFeatureDetector corner = FactoryDetectPoint.createShiTomasi(new ConfigGeneralDetector(1000, 5, 1), false, derivType);
    InterestPointDetector detector = FactoryInterestPoint.wrapPoint(corner, 1, imageType, derivType);
    DescribeRegionPoint describe = FactoryDescribeRegionPoint.brief(new ConfigBrief(true), imageType);
    FactoryDetectDescribe.fuseTogether(detector, null, describe);
  }
}
