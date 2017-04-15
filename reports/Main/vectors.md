Code from [Main.scala:97](../../src/test/scala/Main.scala#L97) executed in 0.11 seconds: 
```java
    FactoryDetectLineAlgs.houghPolar(new ConfigHoughPolar(localMaxRadius, minCounts, 2, resolutionAngle, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
```

Returns: 
```
    boofcv.abst.feature.detect.line.DetectLineHoughPolar@7e5d9a50
```
Code from [Main.scala:61](../../src/test/scala/Main.scala#L61) executed in 2.50 seconds: 
```java
    val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
    gfx.drawImage(image1, 0, 0, width, height, null)
    found.asScala.foreach(line ⇒ {
      if(Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
        val x1 = 0
        val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
        val x2 = image1.getWidth
        val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.RED)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      } else {
        val y1 = 0
        val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
        val y2 = image1.getHeight
        val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.GREEN)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      }
    })
```
Logging: 
```
    LineParametric2D_F32 P( 902.4794 1211.1199 ) Slope( -0.03489945 -0.99939084 ) -> (860,0)->(941,2340)
    LineParametric2D_F32 P( 884.4868 1211.7482 ) Slope( -0.03489945 -0.99939084 ) -> (842,0)->(923,2340)
    LineParametric2D_F32 P( 874.4907 1212.0973 ) Slope( -0.03489945 -0.99939084 ) -> (832,0)->(913,2340)
    LineParametric2D_F32 P( 2160.1997 253.31549 ) Slope( -0.99619466 -0.087155886 ) -> (0,64)->(4160,427)
    LineParametric2D_F32 P( 3040.0479 1186.7577 ) Slope( -0.017452406 0.9998477 ) -> (3060,0)->(3020,2340)
    LineParametric2D_F32 P( 2226.936 2003.3159 ) Slope( -0.9848078 0.1736481 ) -> (0,2395)->(4160,1662)
    LineParametric2D_F32 P( 2158.1074 277.229 ) Slope( -0.99619466 -0.087155886 ) -> (0,88)->(4160,451)
    LineParametric2D_F32 P( 3060.049 1187.1068 ) Slope( -0.017452406 0.9998477 ) -> (3080,0)->(3040,2340)
    LineParametric2D_F32 P( 3052.0486 1186.9672 ) Slope( -0.017452406 0.9998477 ) -> (3072,0)->(3032,2340)
    LineParametric2D_F32 P( 2233.8833 2042.7163 ) Slope( -0.9848078 0.1736481 ) -> (0,2436)->(4160,1703)
    LineParametric2D_F32 P( 2230.0623 2021.0461 ) Slope( -0.9848078 0.1736481 ) -> (0,2414)->(4160,1681)
    LineParametric2D_F32 P( 3843.9365 1262.444 ) Slope( -0.05233596 0.9986295 ) -> (3910,0)->(3788,2340)
    LineParametric2D_F32 P( 3853.9248 1262.9675 ) Slope( -0.05233596 0.9986295 ) -> (3920,0)->(3798,2340)
    LineParametric2D_F32 P( 46.82776 1099.0001 ) Slope( -0.034899496 0.99939084 ) -> (85,0)->(4,2340)
    LineParametric2D_F32 P( 80.33923 1065.2021 ) Slope( -0.05233596 0.9986295 ) -> (136,0)->(14,2340)
    LineParametric2D_F32 P( 2067.663 1052.6227 ) Slope( -0.9945219 0.10452842 ) -> (0,1269)->(4160,832)
    LineParametric2D_F32 P( 156.78296 1237.1602 ) Slope( -0.03489945 -0.99939084 ) -> (113,0)->(194,2340)
    LineParametric2D_F32 P( 200.09094 905.796 ) Slope( -0.1391731 0.99026805 ) -> (327,0)->(-1,2340)
    LineParametric2D_F32 P( 2095.082 305.9568 ) Slope( -0.9998477 -0.017452471 ) -> (0,269)->(4160,341)
    LineParametric2D_F32 P( 221.88135 1431.1417 ) Slope( -0.13917318 -0.99026805 ) -> (20,0)->(348,2340)
    
```

Returns: 
![Result](vectors.0.png)
Code from [Main.scala:100](../../src/test/scala/Main.scala#L100) executed in 0.01 seconds: 
```java
    FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
```

Returns: 
```
    boofcv.abst.feature.detect.line.DetectLineHoughFoot@3dddefd8
```
Code from [Main.scala:61](../../src/test/scala/Main.scala#L61) executed in 0.59 seconds: 
```java
    val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
    gfx.drawImage(image1, 0, 0, width, height, null)
    found.asScala.foreach(line ⇒ {
      if(Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
        val x1 = 0
        val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
        val x2 = image1.getWidth
        val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.RED)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      } else {
        val y1 = 0
        val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
        val y2 = image1.getHeight
        val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.GREEN)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      }
    })
```
Logging: 
```
    LineParametric2D_F32 P( 2094.0 1170.0 ) Slope( 0.0 14.0 ) -> (2094,0)->(2094,2340)
    LineParametric2D_F32 P( 3057.0 1170.0 ) Slope( 0.0 977.0 ) -> (3057,0)->(3057,2340)
    LineParametric2D_F32 P( 2005.0 1174.0 ) Slope( -4.0 -75.0 ) -> (1942,0)->(2066,2340)
    LineParametric2D_F32 P( 3045.0 1170.0 ) Slope( 0.0 965.0 ) -> (3045,0)->(3045,2340)
    LineParametric2D_F32 P( 885.0 1170.0 ) Slope( 0.0 -1195.0 ) -> (885,0)->(885,2340)
    LineParametric2D_F32 P( 898.0 1170.0 ) Slope( 0.0 -1182.0 ) -> (898,0)->(898,2340)
    LineParametric2D_F32 P( 2080.0 1082.0 ) Slope( 88.0 0.0 ) -> (0,1082)->(4160,1082)
    LineParametric2D_F32 P( 2046.0 1180.0 ) Slope( -10.0 -34.0 ) -> (1698,0)->(2386,2340)
    LineParametric2D_F32 P( 874.0 1170.0 ) Slope( 0.0 -1206.0 ) -> (874,0)->(874,2340)
    LineParametric2D_F32 P( 3034.0 1170.0 ) Slope( 0.0 954.0 ) -> (3034,0)->(3034,2340)
    LineParametric2D_F32 P( 3075.0 1170.0 ) Slope( 0.0 995.0 ) -> (3075,0)->(3075,2340)
    LineParametric2D_F32 P( 1940.0 1208.0 ) Slope( -38.0 -140.0 ) -> (1612,0)->(2247,2340)
    LineParametric2D_F32 P( 2084.0 1166.0 ) Slope( 4.0 4.0 ) -> (918,0)->(3258,2340)
    LineParametric2D_F32 P( 2195.0 1165.0 ) Slope( 5.0 115.0 ) -> (2144,0)->(2245,2340)
    LineParametric2D_F32 P( 2153.0 284.0 ) Slope( 886.0 73.0 ) -> (0,106)->(4160,448)
    LineParametric2D_F32 P( 913.0 1170.0 ) Slope( 0.0 -1167.0 ) -> (913,0)->(913,2340)
    LineParametric2D_F32 P( 2249.0 2031.0 ) Slope( -861.0 169.0 ) -> (0,2472)->(4160,1656)
    
```

Returns: 
![Result](vectors.1.png)
Code from [Main.scala:103](../../src/test/scala/Main.scala#L103) executed in 0.00 seconds: 
```java
    FactoryDetectLineAlgs.houghFootSub(new ConfigHoughFootSubimage(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines, totalHorizontalDivisions, totalVerticalDivisions), classOf[GrayU8], classOf[GrayS16])
```

Returns: 
```
    boofcv.abst.feature.detect.line.DetectLineHoughFootSubimage@1d8062d2
```
Code from [Main.scala:61](../../src/test/scala/Main.scala#L61) executed in 0.40 seconds: 
```java
    val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
    gfx.drawImage(image1, 0, 0, width, height, null)
    found.asScala.foreach(line ⇒ {
      if(Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
        val x1 = 0
        val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
        val x2 = image1.getWidth
        val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.RED)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      } else {
        val y1 = 0
        val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
        val y2 = image1.getHeight
        val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
        System.out.println(s"$line -> ($x1,$y1)->($x2,$y2)")
        gfx.setColor(Color.GREEN)
        gfx.drawLine(
          x1 * width / image1.getWidth, y1 * height / image1.getHeight,
          x2 * width / image1.getWidth, y2 * height / image1.getHeight)
      }
    })
```
Logging: 
```
    LineParametric2D_F32 P( 2861.0 1909.0 ) Slope( -8.0 1.0 ) -> (0,2266)->(4160,1746)
    LineParametric2D_F32 P( 3882.0 731.0 ) Slope( 0.0 -18.0 ) -> (3882,0)->(3882,2340)
    LineParametric2D_F32 P( 887.0 1313.0 ) Slope( 3.0 107.0 ) -> (850,0)->(915,2340)
    LineParametric2D_F32 P( 2330.0 270.0 ) Slope( -124.0 -10.0 ) -> (0,82)->(4160,417)
    LineParametric2D_F32 P( 3055.0 1316.0 ) Slope( 0.0 195.0 ) -> (3055,0)->(3055,2340)
    LineParametric2D_F32 P( 1807.0 734.0 ) Slope( -3.0 -13.0 ) -> (1637,0)->(2177,2340)
    LineParametric2D_F32 P( 2386.0 1022.0 ) Slope( 1.0 46.0 ) -> (2363,0)->(2413,2340)
    LineParametric2D_F32 P( 3037.0 1316.0 ) Slope( 0.0 177.0 ) -> (3037,0)->(3037,2340)
    LineParametric2D_F32 P( 905.0 1312.0 ) Slope( 4.0 125.0 ) -> (863,0)->(937,2340)
    LineParametric2D_F32 P( 2337.0 437.0 ) Slope( 1.0 -3.0 ) -> (2482,0)->(1702,2340)
    LineParametric2D_F32 P( 2344.0 1316.0 ) Slope( 0.0 4.0 ) -> (2344,0)->(2344,2340)
    LineParametric2D_F32 P( 3857.0 1316.0 ) Slope( 0.0 -43.0 ) -> (3857,0)->(3857,2340)
    LineParametric2D_F32 P( 1752.0 1023.0 ) Slope( 0.0 -68.0 ) -> (1752,0)->(1752,2340)
    LineParametric2D_F32 P( 1823.0 1083.0 ) Slope( -60.0 3.0 ) -> (0,1174)->(4160,966)
    LineParametric2D_F32 P( 3903.0 1347.0 ) Slope( -31.0 3.0 ) -> (0,1724)->(4160,1322)
    LineParametric2D_F32 P( 1809.0 438.0 ) Slope( 0.0 -11.0 ) -> (1809,0)->(1809,2340)
    LineParametric2D_F32 P( 2340.0 737.0 ) Slope( -6.0 0.0 ) -> (0,737)->(4160,737)
    LineParametric2D_F32 P( 1820.0 446.0 ) Slope( -8.0 0.0 ) -> (0,446)->(4160,446)
    LineParametric2D_F32 P( 3869.0 731.0 ) Slope( 0.0 -31.0 ) -> (3869,0)->(3869,2340)
    LineParametric2D_F32 P( 265.0 438.0 ) Slope( 0.0 5.0 ) -> (265,0)->(265,2340)
    
```

Returns: 
![Result](vectors.2.png)
