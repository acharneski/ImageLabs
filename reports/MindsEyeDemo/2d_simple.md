## Linear
Code from [MindsEyeDemo.scala:284](../../src/test/scala/MindsEyeDemo.scala#L284) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if (x < y) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:193](../../src/test/scala/MindsEyeDemo.scala#L193) executed in 0.54 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    // model = model.add(new MinMaxFilterLayer());
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e5400000001",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e5400000004"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e5400000003",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e5400000002",
          "weights": "[ [ -0.0,0.0 ],[ -0.0,-0.0 ] ]"
        },
        "prev0": {
          "target": "[00395939-8e5a-41e0-ad26-e3ee35ca39a8, 742939af-a98e-47cb-9c13-a376a4a697f0]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:215](../../src/test/scala/MindsEyeDemo.scala#L215) executed in 0.02 seconds: 
```java
    val trainingNetwork: DAGNetwork = new DAGNetwork
    trainingNetwork.add(model)
    trainingNetwork.addLossComponent(new EntropyLossLayer)
    val gradientTrainer: GradientDescentTrainer = new GradientDescentTrainer
    gradientTrainer.setNet(trainingNetwork)
    gradientTrainer.setData(trainingData.toArray)
    new DynamicRateTrainer(gradientTrainer)
```

Returns: 

```
    com.simiacryptus.mindseye.training.DynamicRateTrainer@18b0930f
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.50 seconds: 
```java
    val trainingContext = new TrainingContext
    trainingContext.terminalErr = 0.05
    trainer.step(trainingContext)
    val finalError = trainer.step(trainingContext).finalError
    System.out.println(s"Final Error = $finalError")
    model
```
Logging: 
```
    Final Error = 0.04943143196880331
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e5400000001",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e5400000004"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e5400000003",
        "bias": "[-0.16242782072806472, 0.16242782072805742]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e5400000002",
          "weights": "[ [ -7.4421713967561995,8.07621273706881 ],[ 7.426473005725838,-8.08051412299583 ] ]"
        },
        "prev0": {
          "target": "[00395939-8e5a-41e0-ad26-e3ee35ca39a8, 742939af-a98e-47cb-9c13-a376a4a697f0]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.22 seconds: 
```java
    (0 to 400).foreach(x ⇒ (0 to 400).foreach(y ⇒ {
      function((x / 200.0) - 1.0, (y / 200.0) - 1.0) match {
        case 0 ⇒ gfx.setColor(Color.RED)
        case 1 ⇒ gfx.setColor(Color.GREEN)
      }
      gfx.drawRect(x, y, 1, 1)
    }))
    validationData.foreach(testObj ⇒ {
      val row = new util.LinkedHashMap[String, AnyRef]()
      val result = model.eval(testObj(0)).data.head
      (0 until MAX).maxBy(i ⇒ result.get(i)) match {
        case 0 ⇒ gfx.setColor(Color.PINK)
        case 1 ⇒ gfx.setColor(Color.BLUE)
      }
      val xx = testObj(0).get(0) * 200.0 + 200.0
      val yy = testObj(0).get(1) * 200.0 + 200.0
      gfx.drawRect(xx.toInt - 1, yy.toInt - 1, 3, 3)
    })
```

Returns: 

![Result](2d_simple.1.png)



Code from [MindsEyeDemo.scala:255](../../src/test/scala/MindsEyeDemo.scala#L255) executed in 0.04 seconds: 
```java
    validationData.map(testObj ⇒ {
      val result = model.eval(testObj(0)).data.head
      val prediction: Int = (0 until MAX).maxBy(i ⇒ result.get(i))
      val actual: Int = (0 until MAX).maxBy(i ⇒ testObj(1).get(i))
      actual → prediction
    }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
```

Returns: 

```
    Map(1 -> Map(1 -> 44, 0 -> 1), 0 -> Map(1 -> 3, 0 -> 52))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 52 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 1 | 44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.02 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
    }).toMap
```

Returns: 

```
    Map(0 -> 94.54545454545455, 1 -> 97.77777777777777)
```



Code from [MindsEyeDemo.scala:276](../../src/test/scala/MindsEyeDemo.scala#L276) executed in 0.01 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
    }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
```

Returns: 

```
    96.0
```



## XOR
Code from [MindsEyeDemo.scala:289](../../src/test/scala/MindsEyeDemo.scala#L289) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if ((x < 0) ^ (y < 0)) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:193](../../src/test/scala/MindsEyeDemo.scala#L193) executed in 0.00 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    // model = model.add(new MinMaxFilterLayer());
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e5400000007",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e540000000a"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e5400000009",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e5400000008",
          "weights": "[ [ -0.0,0.0 ],[ -0.0,0.0 ] ]"
        },
        "prev0": {
          "target": "[ffc68fe5-c0de-42ab-bc81-600b800589ea, e6ce6234-3880-41ce-bac2-da0e43f9f359]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:215](../../src/test/scala/MindsEyeDemo.scala#L215) executed in 0.00 seconds: 
```java
    val trainingNetwork: DAGNetwork = new DAGNetwork
    trainingNetwork.add(model)
    trainingNetwork.addLossComponent(new EntropyLossLayer)
    val gradientTrainer: GradientDescentTrainer = new GradientDescentTrainer
    gradientTrainer.setNet(trainingNetwork)
    gradientTrainer.setData(trainingData.toArray)
    new DynamicRateTrainer(gradientTrainer)
```

Returns: 

```
    com.simiacryptus.mindseye.training.DynamicRateTrainer@4d0402b
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.44 seconds: 
```java
    val trainingContext = new TrainingContext
    trainingContext.terminalErr = 0.05
    trainer.step(trainingContext)
    val finalError = trainer.step(trainingContext).finalError
    System.out.println(s"Final Error = $finalError")
    model
```
Logging: 
```
    Final Error = 0.6715853105996983
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e5400000007",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e540000000a"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e5400000009",
        "bias": "[0.12474083063971575, -0.12474083063971562]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e5400000008",
          "weights": "[ [ -0.2992478178440789,-0.08017059331556768 ],[ 0.2785387650026395,0.06150159446849259 ] ]"
        },
        "prev0": {
          "target": "[ffc68fe5-c0de-42ab-bc81-600b800589ea, e6ce6234-3880-41ce-bac2-da0e43f9f359]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.09 seconds: 
```java
    (0 to 400).foreach(x ⇒ (0 to 400).foreach(y ⇒ {
      function((x / 200.0) - 1.0, (y / 200.0) - 1.0) match {
        case 0 ⇒ gfx.setColor(Color.RED)
        case 1 ⇒ gfx.setColor(Color.GREEN)
      }
      gfx.drawRect(x, y, 1, 1)
    }))
    validationData.foreach(testObj ⇒ {
      val row = new util.LinkedHashMap[String, AnyRef]()
      val result = model.eval(testObj(0)).data.head
      (0 until MAX).maxBy(i ⇒ result.get(i)) match {
        case 0 ⇒ gfx.setColor(Color.PINK)
        case 1 ⇒ gfx.setColor(Color.BLUE)
      }
      val xx = testObj(0).get(0) * 200.0 + 200.0
      val yy = testObj(0).get(1) * 200.0 + 200.0
      gfx.drawRect(xx.toInt - 1, yy.toInt - 1, 3, 3)
    })
```

Returns: 

![Result](2d_simple.2.png)



Code from [MindsEyeDemo.scala:255](../../src/test/scala/MindsEyeDemo.scala#L255) executed in 0.01 seconds: 
```java
    validationData.map(testObj ⇒ {
      val result = model.eval(testObj(0)).data.head
      val prediction: Int = (0 until MAX).maxBy(i ⇒ result.get(i))
      val actual: Int = (0 until MAX).maxBy(i ⇒ testObj(1).get(i))
      actual → prediction
    }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
```

Returns: 

```
    Map(1 -> Map(1 -> 13, 0 -> 43), 0 -> Map(1 -> 11, 0 -> 33))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 33 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 43 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
    }).toMap
```

Returns: 

```
    Map(0 -> 75.0, 1 -> 23.214285714285715)
```



Code from [MindsEyeDemo.scala:276](../../src/test/scala/MindsEyeDemo.scala#L276) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
    }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
```

Returns: 

```
    46.0
```



## Circle
Code from [MindsEyeDemo.scala:294](../../src/test/scala/MindsEyeDemo.scala#L294) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if ((x * x) + (y * y) < 0.5) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:193](../../src/test/scala/MindsEyeDemo.scala#L193) executed in 0.00 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    // model = model.add(new MinMaxFilterLayer());
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e540000000d",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e5400000010"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e540000000f",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e540000000e",
          "weights": "[ [ 0.0,-0.0 ],[ -0.0,0.0 ] ]"
        },
        "prev0": {
          "target": "[b1b25bfb-77e3-4694-a15d-6eefabd48c88, 93b59930-9627-4324-90f0-6bb05ff65162]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:215](../../src/test/scala/MindsEyeDemo.scala#L215) executed in 0.00 seconds: 
```java
    val trainingNetwork: DAGNetwork = new DAGNetwork
    trainingNetwork.add(model)
    trainingNetwork.addLossComponent(new EntropyLossLayer)
    val gradientTrainer: GradientDescentTrainer = new GradientDescentTrainer
    gradientTrainer.setNet(trainingNetwork)
    gradientTrainer.setData(trainingData.toArray)
    new DynamicRateTrainer(gradientTrainer)
```

Returns: 

```
    com.simiacryptus.mindseye.training.DynamicRateTrainer@2577d6c8
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.50 seconds: 
```java
    val trainingContext = new TrainingContext
    trainingContext.terminalErr = 0.05
    trainer.step(trainingContext)
    val finalError = trainer.step(trainingContext).finalError
    System.out.println(s"Final Error = $finalError")
    model
```
Logging: 
```
    Final Error = 0.6841059804258929
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "11e92e40-a158-47de-9b81-1e540000000d",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "11e92e40-a158-47de-9b81-1e5400000010"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "11e92e40-a158-47de-9b81-1e540000000f",
        "bias": "[-0.11474012007895591, 0.11474012007895623]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "11e92e40-a158-47de-9b81-1e540000000e",
          "weights": "[ [ -0.017574447670072112,-0.10280519355887682 ],[ 0.025321079720268678,0.1027261295833255 ] ]"
        },
        "prev0": {
          "target": "[b1b25bfb-77e3-4694-a15d-6eefabd48c88, 93b59930-9627-4324-90f0-6bb05ff65162]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.10 seconds: 
```java
    (0 to 400).foreach(x ⇒ (0 to 400).foreach(y ⇒ {
      function((x / 200.0) - 1.0, (y / 200.0) - 1.0) match {
        case 0 ⇒ gfx.setColor(Color.RED)
        case 1 ⇒ gfx.setColor(Color.GREEN)
      }
      gfx.drawRect(x, y, 1, 1)
    }))
    validationData.foreach(testObj ⇒ {
      val row = new util.LinkedHashMap[String, AnyRef]()
      val result = model.eval(testObj(0)).data.head
      (0 until MAX).maxBy(i ⇒ result.get(i)) match {
        case 0 ⇒ gfx.setColor(Color.PINK)
        case 1 ⇒ gfx.setColor(Color.BLUE)
      }
      val xx = testObj(0).get(0) * 200.0 + 200.0
      val yy = testObj(0).get(1) * 200.0 + 200.0
      gfx.drawRect(xx.toInt - 1, yy.toInt - 1, 3, 3)
    })
```

Returns: 

![Result](2d_simple.3.png)



Code from [MindsEyeDemo.scala:255](../../src/test/scala/MindsEyeDemo.scala#L255) executed in 0.00 seconds: 
```java
    validationData.map(testObj ⇒ {
      val result = model.eval(testObj(0)).data.head
      val prediction: Int = (0 until MAX).maxBy(i ⇒ result.get(i))
      val actual: Int = (0 until MAX).maxBy(i ⇒ testObj(1).get(i))
      actual → prediction
    }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
```

Returns: 

```
    Map(1 -> Map(1 -> 71), 0 -> Map(1 -> 29))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 0 | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 0 | 71 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
    }).toMap
```

Returns: 

```
    Map(0 -> 0.0, 1 -> 100.0)
```



Code from [MindsEyeDemo.scala:276](../../src/test/scala/MindsEyeDemo.scala#L276) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
    }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
```

Returns: 

```
    71.0
```



