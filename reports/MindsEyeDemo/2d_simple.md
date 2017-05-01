## Linear
Code from [MindsEyeDemo.scala:272](../../src/test/scala/MindsEyeDemo.scala#L272) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if (x < y) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:274](../../src/test/scala/MindsEyeDemo.scala#L274) executed in 0.00 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.1
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "DAGNetwork",
      "id": "bda0cb2d-e331-453b-937e-ddad00000007",
      "root": {
        "layer": {
          "class": "SoftmaxActivationLayer",
          "id": "bda0cb2d-e331-453b-937e-ddad0000000a"
        },
        "prev0": {
          "layer": {
            "class": "BiasLayer",
            "id": "bda0cb2d-e331-453b-937e-ddad00000009",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "layer": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bda0cb2d-e331-453b-937e-ddad00000008",
              "weights": "[ [ 0.06448892747039674,0.004231648714995081 ],[ -0.09592902742583825,-0.08202095096615354 ] ]"
            },
            "prev0": {
              "target": "[8377ca1f-7c6f-46ff-872b-125737c6c0fb, c7d833c5-0e26-4864-97fb-592fa5352c02]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:248](../../src/test/scala/MindsEyeDemo.scala#L248) executed in 0.11 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.1.png)



Code from [MindsEyeDemo.scala:265](../../src/test/scala/MindsEyeDemo.scala#L265) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (100.0,Map(0 -> 100.0, 1 -> 100.0))
```



## XOR
Code from [MindsEyeDemo.scala:285](../../src/test/scala/MindsEyeDemo.scala#L285) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if ((x < 0) ^ (y < 0)) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:288](../../src/test/scala/MindsEyeDemo.scala#L288) executed in 0.00 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "DAGNetwork",
      "id": "bda0cb2d-e331-453b-937e-ddad0000000d",
      "root": {
        "layer": {
          "class": "SoftmaxActivationLayer",
          "id": "bda0cb2d-e331-453b-937e-ddad00000010"
        },
        "prev0": {
          "layer": {
            "class": "BiasLayer",
            "id": "bda0cb2d-e331-453b-937e-ddad0000000f",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "layer": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bda0cb2d-e331-453b-937e-ddad0000000e",
              "weights": "[ [ 0.04230207839032445,0.28834874694510737 ],[ 0.0320390778532303,0.039546252345962915 ] ]"
            },
            "prev0": {
              "target": "[c607503c-2832-498e-9212-0c44f4ba4a61, 2585e083-aea7-4cf1-b04c-ce8b84348dd3]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:248](../../src/test/scala/MindsEyeDemo.scala#L248) executed in 0.08 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.2.png)



Code from [MindsEyeDemo.scala:265](../../src/test/scala/MindsEyeDemo.scala#L265) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (49.0,Map(0 -> 100.0, 1 -> 0.0))
```



Code from [MindsEyeDemo.scala:297](../../src/test/scala/MindsEyeDemo.scala#L297) executed in 0.01 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    val middleSize = Array[Int](15)
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), middleSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model = model.add(new BiasLayer(middleSize: _*))
    model = model.add(new AbsActivationLayer())
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(middleSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "DAGNetwork",
      "id": "bda0cb2d-e331-453b-937e-ddad00000013",
      "root": {
        "layer": {
          "class": "SoftmaxActivationLayer",
          "id": "bda0cb2d-e331-453b-937e-ddad00000019"
        },
        "prev0": {
          "layer": {
            "class": "BiasLayer",
            "id": "bda0cb2d-e331-453b-937e-ddad00000018",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "layer": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bda0cb2d-e331-453b-937e-ddad00000017",
              "weights": "[ [ 1.1504511872345788,0.268322812306804 ],[ 0.10153809925131507,-1.5645151601276412 ],[ 0.22832810035500323,0.2944140154786153 ],[ 0.891432915988393,1.3819968037382682 ],[ 0.2280388445363336,0.1502648712995983 ],[ 0.15680070179126593,-0.2937928828855539 ],[ -1.800957527697971,-0.8245570354145833 ],[ 1.128877613374575,0.7576300137489763 ],... ]"
            },
            "prev0": {
              "layer": {
                "class": "AbsActivationLayer",
                "id": "bda0cb2d-e331-453b-937e-ddad00000016"
              },
              "prev0": {
                "layer": {
                  "class": "BiasLayer",
                  "id": "bda0cb2d-e331-453b-937e-ddad00000015",
                  "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                },
                "prev0": {
                  "layer": {
                    "class": "DenseSynapseLayerJBLAS",
                    "id": "bda0cb2d-e331-453b-937e-ddad00000014",
                    "weights": "[ [ 0.6767869237031598,-1.9295589342656005,-1.0309726644062092,0.564611440207325,0.3138618711453875,0.791844371230277,1.1143221707990643,1.2238424243616832,... ],[ -0.5660365609075952,0.18992147579040752,0.3101517357170239,-0.08942412512508476,-1.3699114811262127,-0.7797175117507096,-0.916399771798046,2.0539480585655365,... ] ]"
                  },
                  "prev0": {
                    "target": "[4f8fef50-922d-4dd6-b6cf-add0a1f815be, 73594a1e-e787-4f64-b4ea-07e4758712c4]"
                  }
                }
              }
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:248](../../src/test/scala/MindsEyeDemo.scala#L248) executed in 0.09 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.3.png)



Code from [MindsEyeDemo.scala:265](../../src/test/scala/MindsEyeDemo.scala#L265) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (99.0,Map(0 -> 100.0, 1 -> 97.67441860465117))
```



## Circle
Code from [MindsEyeDemo.scala:314](../../src/test/scala/MindsEyeDemo.scala#L314) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if ((x * x) + (y * y) < 0.5) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:317](../../src/test/scala/MindsEyeDemo.scala#L317) executed in 0.00 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "DAGNetwork",
      "id": "bda0cb2d-e331-453b-937e-ddad0000001c",
      "root": {
        "layer": {
          "class": "SoftmaxActivationLayer",
          "id": "bda0cb2d-e331-453b-937e-ddad0000001f"
        },
        "prev0": {
          "layer": {
            "class": "BiasLayer",
            "id": "bda0cb2d-e331-453b-937e-ddad0000001e",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "layer": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bda0cb2d-e331-453b-937e-ddad0000001d",
              "weights": "[ [ -0.003356582745107412,0.0249142274308162 ],[ -0.2928900855078193,-0.18830722067882058 ] ]"
            },
            "prev0": {
              "target": "[706aed6a-09d0-4d25-8a32-3a40212e5f55, d1160941-50fb-4213-849e-1bee04f6fdd7]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:248](../../src/test/scala/MindsEyeDemo.scala#L248) executed in 0.10 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.4.png)



Code from [MindsEyeDemo.scala:265](../../src/test/scala/MindsEyeDemo.scala#L265) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (27.0,Map(0 -> 100.0, 1 -> 0.0))
```



Code from [MindsEyeDemo.scala:326](../../src/test/scala/MindsEyeDemo.scala#L326) executed in 0.01 seconds: 
```java
    var model: DAGNetwork = new DAGNetwork
    val middleSize = Array[Int](15)
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), middleSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model = model.add(new BiasLayer(middleSize: _*))
    model = model.add(new AbsActivationLayer())
    model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(middleSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model = model.add(new BiasLayer(outputSize: _*))
    model = model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "DAGNetwork",
      "id": "bda0cb2d-e331-453b-937e-ddad00000022",
      "root": {
        "layer": {
          "class": "SoftmaxActivationLayer",
          "id": "bda0cb2d-e331-453b-937e-ddad00000028"
        },
        "prev0": {
          "layer": {
            "class": "BiasLayer",
            "id": "bda0cb2d-e331-453b-937e-ddad00000027",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "layer": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bda0cb2d-e331-453b-937e-ddad00000026",
              "weights": "[ [ -0.4961551027188386,-1.6025524003592786 ],[ 0.6767292416578095,-0.8863451967712533 ],[ -1.121203241198936,-0.3333470394557084 ],[ 0.9733358170048696,0.6715177421394439 ],[ 0.3229363904020222,-0.9522514379734052 ],[ -0.32358270664361916,-1.2090140024843026 ],[ -0.8040364686228791,-1.5256932878113028 ],[ 0.7149060426707465,1.0272808664745223 ],... ]"
            },
            "prev0": {
              "layer": {
                "class": "AbsActivationLayer",
                "id": "bda0cb2d-e331-453b-937e-ddad00000025"
              },
              "prev0": {
                "layer": {
                  "class": "BiasLayer",
                  "id": "bda0cb2d-e331-453b-937e-ddad00000024",
                  "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                },
                "prev0": {
                  "layer": {
                    "class": "DenseSynapseLayerJBLAS",
                    "id": "bda0cb2d-e331-453b-937e-ddad00000023",
                    "weights": "[ [ 0.9625768457868111,-0.38876680380037737,-0.4523328529612256,-1.0216622841109824,0.8449977001476421,0.45399525572378346,-2.050719341712142,-0.013319889855142665,... ],[ 1.3616539843264095,0.09302535608072968,-2.2490902181775385,-1.1015821196548041,-1.2003008029304807,0.9784597150839556,0.30476902633282654,0.2599163782189681,... ] ]"
                  },
                  "prev0": {
                    "target": "[fef293ab-0111-4b63-8719-6af62291d339, 4b2d9e56-dfb1-4088-a461-8e3871b9cb43]"
                  }
                }
              }
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:248](../../src/test/scala/MindsEyeDemo.scala#L248) executed in 0.08 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.5.png)



Code from [MindsEyeDemo.scala:265](../../src/test/scala/MindsEyeDemo.scala#L265) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (98.0,Map(0 -> 95.83333333333333, 1 -> 100.0))
```



