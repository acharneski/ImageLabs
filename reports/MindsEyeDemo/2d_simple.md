## Linear
Code from [MindsEyeDemo.scala:284](../../src/test/scala/MindsEyeDemo.scala#L284) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if (x < y) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:193](../../src/test/scala/MindsEyeDemo.scala#L193) executed in 0.43 seconds: 
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
  "id": "a4dda4ba-25d5-4841-afcb-eaf300000001",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf300000004"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf300000003",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf300000002",
          "weights": "[ [ 0.0,-0.0 ],[ -0.0,0.0 ] ]"
        },
        "prev0": {
          "target": "[eecedbbf-899c-473b-954e-4eb7288db3d3, 877226ce-00a3-43ad-839e-962ec0bcf5f7]"
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
    com.simiacryptus.mindseye.training.DynamicRateTrainer@dfddc9a
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.39 seconds: 
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
    Final Error = 0.0490268634513138
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "a4dda4ba-25d5-4841-afcb-eaf300000001",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf300000004"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf300000003",
        "bias": "[0.29512984036659345, -0.29512984036659323]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf300000002",
          "weights": "[ [ -6.2172567586352,6.613935986802884 ],[ 6.225318307156802,-6.608038259837675 ] ]"
        },
        "prev0": {
          "target": "[eecedbbf-899c-473b-954e-4eb7288db3d3, 877226ce-00a3-43ad-839e-962ec0bcf5f7]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.20 seconds: 
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



Code from [MindsEyeDemo.scala:255](../../src/test/scala/MindsEyeDemo.scala#L255) executed in 0.05 seconds: 
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
    Map(1 -> Map(1 -> 43, 0 -> 3), 0 -> Map(0 -> 54))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 3 | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.02 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
    }).toMap
```

Returns: 

```
    Map(0 -> 100.0, 1 -> 93.47826086956522)
```



Code from [MindsEyeDemo.scala:276](../../src/test/scala/MindsEyeDemo.scala#L276) executed in 0.01 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
    }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
```

Returns: 

```
    97.0
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
  "id": "a4dda4ba-25d5-4841-afcb-eaf300000007",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf30000000a"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf300000009",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf300000008",
          "weights": "[ [ -0.0,0.0 ],[ 0.0,0.0 ] ]"
        },
        "prev0": {
          "target": "[0d880976-3b26-48da-a3ef-6ed9da885f04, 52f6e1dc-aeb6-4b76-9e05-d3670cb76cd2]"
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
    com.simiacryptus.mindseye.training.DynamicRateTrainer@36ac8a63
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.53 seconds: 
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
    Final Error = 0.685246796510372
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "a4dda4ba-25d5-4841-afcb-eaf300000007",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf30000000a"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf300000009",
        "bias": "[0.08737132856292004, -0.08737132856292022]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf300000008",
          "weights": "[ [ 0.17034694961902874,0.046238980953907015 ],[ -0.1660355675659233,-0.045384566245163606 ] ]"
        },
        "prev0": {
          "target": "[0d880976-3b26-48da-a3ef-6ed9da885f04, 52f6e1dc-aeb6-4b76-9e05-d3670cb76cd2]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.13 seconds: 
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
    Map(1 -> Map(1 -> 19, 0 -> 32), 0 -> Map(1 -> 10, 0 -> 39))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 39 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 32 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
    }).toMap
```

Returns: 

```
    Map(0 -> 79.59183673469387, 1 -> 37.254901960784316)
```



Code from [MindsEyeDemo.scala:276](../../src/test/scala/MindsEyeDemo.scala#L276) executed in 0.00 seconds: 
```java
    (0 until MAX).map(actual ⇒ {
      categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
    }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
```

Returns: 

```
    58.0
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
  "id": "a4dda4ba-25d5-4841-afcb-eaf30000000d",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf300000010"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf30000000f",
        "bias": "[0.0, 0.0]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf30000000e",
          "weights": "[ [ -0.0,-0.0 ],[ 0.0,0.0 ] ]"
        },
        "prev0": {
          "target": "[dff1517f-fa94-44d3-8608-266152976f5d, 414a3040-44d6-47e7-8fee-9c30a69a3db6]"
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
    com.simiacryptus.mindseye.training.DynamicRateTrainer@15b986cd
```



Code from [MindsEyeDemo.scala:225](../../src/test/scala/MindsEyeDemo.scala#L225) executed in 0.31 seconds: 
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
    Final Error = 0.633266086910647
    
```

Returns: 

```
    {
  "class": "DAGNetwork",
  "id": "a4dda4ba-25d5-4841-afcb-eaf30000000d",
  "root": {
    "layer": {
      "class": "SoftmaxActivationLayer",
      "id": "a4dda4ba-25d5-4841-afcb-eaf300000010"
    },
    "prev0": {
      "layer": {
        "class": "BiasLayer",
        "id": "a4dda4ba-25d5-4841-afcb-eaf30000000f",
        "bias": "[-0.35217301215882385, 0.3521730121588237]"
      },
      "prev0": {
        "layer": {
          "class": "DenseSynapseLayerJBLAS",
          "id": "a4dda4ba-25d5-4841-afcb-eaf30000000e",
          "weights": "[ [ -0.04244980909725282,0.06143960879283192 ],[ 0.04178079178131982,-0.06144655096671015 ] ]"
        },
        "prev0": {
          "target": "[dff1517f-fa94-44d3-8608-266152976f5d, 414a3040-44d6-47e7-8fee-9c30a69a3db6]"
        }
      }
    }
  }
}
```



Code from [MindsEyeDemo.scala:234](../../src/test/scala/MindsEyeDemo.scala#L234) executed in 0.13 seconds: 
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
    Map(1 -> Map(1 -> 58), 0 -> Map(1 -> 42))
```



Actual \ Predicted | 0 | 1
--- | --- | ---
 **0** | 0 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
 **1** | 0 | 58 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0

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
    58.0
```



