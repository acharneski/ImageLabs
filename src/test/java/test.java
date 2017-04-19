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
        GeneralFeatureDetector corner = FactoryDetectPoint.createShiTomasi(new ConfigGeneralDetector(1000,5,1), false, derivType);
        InterestPointDetector detector = FactoryInterestPoint.wrapPoint(corner, 1, imageType, derivType);
        DescribeRegionPoint describe = FactoryDescribeRegionPoint.brief(new ConfigBrief(true), imageType);
        FactoryDetectDescribe.fuseTogether(detector, null, describe);
    }
}
