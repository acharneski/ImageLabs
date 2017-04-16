Code from [BoofcvSpec.scala:140](../../src/test/scala/BoofcvSpec.scala#L140) executed in 0.01 seconds: 
```java
    FactoryDetectLineAlgs.lineRansac(40, 30, 2.36, true, classOf[GrayF32], classOf[GrayF32])
```

Returns: 
```
    boofcv.abst.feature.detect.line.DetectLineSegmentsGridRansac@46f699d5
```
Code from [BoofcvSpec.scala:121](../../src/test/scala/BoofcvSpec.scala#L121) executed in 1.64 seconds: 
```java
    val found: util.List[LineSegment2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayF32]))
    gfx.drawImage(image1, 0, 0, width, height, null)
    gfx.setColor(Color.GREEN)
    found.asScala.foreach(line ⇒ {
      gfx.drawLine(
        (line.a.x * width / image1.getWidth).toInt, (line.a.y * height / image1.getHeight).toInt,
        (line.b.x * width / image1.getWidth).toInt, (line.b.y * height / image1.getHeight).toInt)
    })
```

Returns: 
![Result](segments.0.png)
