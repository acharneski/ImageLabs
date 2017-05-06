## Linear
The simplest problem is linear descrimination, which can be learned by the simplest network

Code from [MindsEyeDemo.scala:271](../../src/test/scala/MindsEyeDemo.scala#L271) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if (x < y) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:273](../../src/test/scala/MindsEyeDemo.scala#L273) executed in 0.00 seconds: 
```java
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new DenseSynapseLayerJBLAS(inputSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.1
    }))
    model.add(new BiasLayer(outputSize: _*))
    model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "PipelineNetwork",
      "id": "bf91e8ca-00fc-4e91-bc12-526300000007",
      "nodes": [
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000008",
            "weights": "[ [ -0.03636484932721239,1.2188257931298182E-4 ],[ 0.0111983680432676,0.023474488635533062 ] ]"
          },
          "prev0": {
            "target": "[84015669-8072-4e31-981e-54299d95f683]"
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000009",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000008",
              "weights": "[ [ -0.03636484932721239,1.2188257931298182E-4 ],[ 0.0111983680432676,0.023474488635533062 ] ]"
            },
            "prev0": {
              "target": "[84015669-8072-4e31-981e-54299d95f683]"
            }
          }
        },
        {
          "id": {
            "class": "SoftmaxActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000000a"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000009",
              "bias": "[0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000008",
                "weights": "[ [ -0.03636484932721239,1.2188257931298182E-4 ],[ 0.0111983680432676,0.023474488635533062 ] ]"
              },
              "prev0": {
                "target": "[84015669-8072-4e31-981e-54299d95f683]"
              }
            }
          }
        }
      ],
      "root": {
        "id": {
          "class": "SoftmaxActivationLayer",
          "id": "bf91e8ca-00fc-4e91-bc12-52630000000a"
        },
        "prev0": {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000009",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000008",
              "weights": "[ [ -0.03636484932721239,1.2188257931298182E-4 ],[ 0.0111983680432676,0.023474488635533062 ] ]"
            },
            "prev0": {
              "target": "[84015669-8072-4e31-981e-54299d95f683]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:214](../../src/test/scala/MindsEyeDemo.scala#L214) executed in 0.00 seconds: 
```java
    val trainingNetwork: SupervisedNetwork = new SupervisedNetwork(model, new EntropyLossLayer)
    val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setTimeout(10, TimeUnit.SECONDS)
    trainer.setTerminateThreshold(0.0)
    trainer
```

Returns: 

```
    com.simiacryptus.mindseye.opt.IterativeTrainer@59f7c106
```



Code from [MindsEyeDemo.scala:222](../../src/test/scala/MindsEyeDemo.scala#L222) executed in 10.01 seconds: 
```java
    trainer.run()
```

Returns: 

```
    NaN
```



Code from [MindsEyeDemo.scala:246](../../src/test/scala/MindsEyeDemo.scala#L246) executed in 0.10 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.1.png)



Code from [MindsEyeDemo.scala:263](../../src/test/scala/MindsEyeDemo.scala#L263) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (95.0,Map(0 -> 88.88888888888889, 1 -> 100.0))
```



## XOR
The XOR function is not linearly seperable, and cannot be solved by this network:

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
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new DenseSynapseLayerJBLAS(inputSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
    }))
    model.add(new BiasLayer(outputSize: _*))
    model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "PipelineNetwork",
      "id": "bf91e8ca-00fc-4e91-bc12-52630000000d",
      "nodes": [
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000000e",
            "weights": "[ [ -0.4385123429203249,-0.154908578595438 ],[ 0.33273099198612877,8.357791682557198E-4 ] ]"
          },
          "prev0": {
            "target": "[12f54a38-0e41-475b-b59b-89ffcfb6144b]"
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000000f",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000000e",
              "weights": "[ [ -0.4385123429203249,-0.154908578595438 ],[ 0.33273099198612877,8.357791682557198E-4 ] ]"
            },
            "prev0": {
              "target": "[12f54a38-0e41-475b-b59b-89ffcfb6144b]"
            }
          }
        },
        {
          "id": {
            "class": "SoftmaxActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000010"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000000f",
              "bias": "[0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-52630000000e",
                "weights": "[ [ -0.4385123429203249,-0.154908578595438 ],[ 0.33273099198612877,8.357791682557198E-4 ] ]"
              },
              "prev0": {
                "target": "[12f54a38-0e41-475b-b59b-89ffcfb6144b]"
              }
            }
          }
        }
      ],
      "root": {
        "id": {
          "class": "SoftmaxActivationLayer",
          "id": "bf91e8ca-00fc-4e91-bc12-526300000010"
        },
        "prev0": {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000000f",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000000e",
              "weights": "[ [ -0.4385123429203249,-0.154908578595438 ],[ 0.33273099198612877,8.357791682557198E-4 ] ]"
            },
            "prev0": {
              "target": "[12f54a38-0e41-475b-b59b-89ffcfb6144b]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:214](../../src/test/scala/MindsEyeDemo.scala#L214) executed in 0.00 seconds: 
```java
    val trainingNetwork: SupervisedNetwork = new SupervisedNetwork(model, new EntropyLossLayer)
    val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setTimeout(10, TimeUnit.SECONDS)
    trainer.setTerminateThreshold(0.0)
    trainer
```

Returns: 

```
    com.simiacryptus.mindseye.opt.IterativeTrainer@1ec4fe38
```



Code from [MindsEyeDemo.scala:222](../../src/test/scala/MindsEyeDemo.scala#L222) executed in 10.00 seconds: 
```java
    trainer.run()
```

Returns: 

```
    NaN
```



Code from [MindsEyeDemo.scala:246](../../src/test/scala/MindsEyeDemo.scala#L246) executed in 0.09 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.2.png)



Code from [MindsEyeDemo.scala:263](../../src/test/scala/MindsEyeDemo.scala#L263) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (59.0,Map(0 -> 48.83720930232558, 1 -> 66.66666666666667))
```



If we add a hidden id with enough units, we can learn the nonlinearity:

Code from [MindsEyeDemo.scala:298](../../src/test/scala/MindsEyeDemo.scala#L298) executed in 0.01 seconds: 
```java
    var model: PipelineNetwork = new PipelineNetwork
    val middleSize = Array[Int](15)
    model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model.add(new BiasLayer(middleSize: _*))
    model.add(new AbsActivationLayer())
    model.add(new DenseSynapseLayerJBLAS(middleSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model.add(new BiasLayer(outputSize: _*))
    model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "PipelineNetwork",
      "id": "bf91e8ca-00fc-4e91-bc12-526300000013",
      "nodes": [
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
            "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
          },
          "prev0": {
            "target": "[011ef31c-0eac-4e49-b7db-50e39c725621]"
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000015",
            "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
              "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
            },
            "prev0": {
              "target": "[011ef31c-0eac-4e49-b7db-50e39c725621]"
            }
          }
        },
        {
          "id": {
            "class": "AbsActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000016"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000015",
              "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
                "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
              },
              "prev0": {
                "target": "[011ef31c-0eac-4e49-b7db-50e39c725621]"
              }
            }
          }
        },
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000017",
            "weights": "[ [ 1.1743034034795576,-1.5298001946772113 ],[ 1.3685270672090621,0.9036747561425088 ],[ 0.2904167499816018,0.23939079150039738 ],[ -0.8249147237333315,0.09301636545663848 ],[ 1.0369888199133972,-0.6057178690991806 ],[ -0.7454841166563169,-0.12880925746491018 ],[ -0.3485707523631505,1.7468276941981997 ],[ 0.051788283507919024,-0.12513386702176343 ],... ]"
          },
          "prev0": {
            "id": {
              "class": "AbsActivationLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000016"
            },
            "prev0": {
              "id": {
                "class": "BiasLayer",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000015",
                "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
              },
              "prev0": {
                "id": {
                  "class": "DenseSynapseLayerJBLAS",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
                  "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
                },
                "prev0": {
                  "target": "[011ef31c-0eac-4e49-b7db-50e39c725621]"
                }
              }
            }
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000018",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000017",
              "weights": "[ [ 1.1743034034795576,-1.5298001946772113 ],[ 1.3685270672090621,0.9036747561425088 ],[ 0.2904167499816018,0.23939079150039738 ],[ -0.8249147237333315,0.09301636545663848 ],[ 1.0369888199133972,-0.6057178690991806 ],[ -0.7454841166563169,-0.12880925746491018 ],[ -0.3485707523631505,1.7468276941981997 ],[ 0.051788283507919024,-0.12513386702176343 ],... ]"
            },
            "prev0": {
              "id": {
                "class": "AbsActivationLayer",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000016"
              },
              "prev0": {
                "id": {
                  "class": "BiasLayer",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000015",
                  "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                },
                "prev0": {
                  "id": {
                    "class": "DenseSynapseLayerJBLAS",
                    "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
                    "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
                  },
                  "prev0": {
                    "target": "[011ef31c-0eac-4e49-b7db-50e39c725621]"
                  }
                }
              }
            }
          }
        },
        {
          "id": {
            "class": "SoftmaxActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000019"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000018",
              "bias": "[0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000017",
                "weights": "[ [ 1.1743034034795576,-1.5298001946772113 ],[ 1.3685270672090621,0.9036747561425088 ],[ 0.2904167499816018,0.23939079150039738 ],[ -0.8249147237333315,0.09301636545663848 ],[ 1.0369888199133972,-0.6057178690991806 ],[ -0.7454841166563169,-0.12880925746491018 ],[ -0.3485707523631505,1.7468276941981997 ],[ 0.051788283507919024,-0.12513386702176343 ],... ]"
              },
              "prev0": {
                "id": {
                  "class": "AbsActivationLayer",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000016"
                },
                "prev0": {
                  "id": {
                    "class": "BiasLayer",
                    "id": "bf91e8ca-00fc-4e91-bc12-526300000015",
                    "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                  },
                  "prev0": {
                    "id": {
                      "class": "DenseSynapseLayerJBLAS",
                      "id": "bf91e8ca-00fc-4e91-bc12-526300000014",
                      "weights": "[ [ -1.2311996971141235,-0.003707473001929643,-2.359939463451168,0.21423755566661767,0.24924599064998587,0.5054974544847066,-0.16626506491687199,0.23599140859545203,... ],[ -0.6308951368661083,-1.2971873749113163,0.2100940366571568,1.6698466625327302,-0.7545029965635098,1.9420542203877156,1.0011124068275807,0.26619129564012767,... ] ]"
    ... and 2325 more bytes
```



Code from [MindsEyeDemo.scala:214](../../src/test/scala/MindsEyeDemo.scala#L214) executed in 0.00 seconds: 
```java
    val trainingNetwork: SupervisedNetwork = new SupervisedNetwork(model, new EntropyLossLayer)
    val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setTimeout(10, TimeUnit.SECONDS)
    trainer.setTerminateThreshold(0.0)
    trainer
```

Returns: 

```
    com.simiacryptus.mindseye.opt.IterativeTrainer@27d0b621
```



Code from [MindsEyeDemo.scala:222](../../src/test/scala/MindsEyeDemo.scala#L222) executed in 10.00 seconds: 
```java
    trainer.run()
```

Returns: 

```
    NaN
```



Code from [MindsEyeDemo.scala:246](../../src/test/scala/MindsEyeDemo.scala#L246) executed in 0.08 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.3.png)



Code from [MindsEyeDemo.scala:263](../../src/test/scala/MindsEyeDemo.scala#L263) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (97.0,Map(0 -> 97.5609756097561, 1 -> 96.61016949152543))
```



## Circle
Similar behavior is seen with simple networks on the unit circle function

Code from [MindsEyeDemo.scala:316](../../src/test/scala/MindsEyeDemo.scala#L316) executed in 0.00 seconds: 
```java
    (x: Double, y: Double) ⇒ if ((x * x) + (y * y) < 0.5) 0 else 1
```

Returns: 

```
    <function2>
```



Code from [MindsEyeDemo.scala:319](../../src/test/scala/MindsEyeDemo.scala#L319) executed in 0.00 seconds: 
```java
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new DenseSynapseLayerJBLAS(inputSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
    }))
    model.add(new BiasLayer(outputSize: _*))
    model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "PipelineNetwork",
      "id": "bf91e8ca-00fc-4e91-bc12-52630000001c",
      "nodes": [
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000001d",
            "weights": "[ [ 0.02161199454033596,-0.05006336570006144 ],[ -0.24400763548472423,-0.055548360446179015 ] ]"
          },
          "prev0": {
            "target": "[e8069e1f-07fb-468e-aac0-5d9f79db1c93]"
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000001e",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000001d",
              "weights": "[ [ 0.02161199454033596,-0.05006336570006144 ],[ -0.24400763548472423,-0.055548360446179015 ] ]"
            },
            "prev0": {
              "target": "[e8069e1f-07fb-468e-aac0-5d9f79db1c93]"
            }
          }
        },
        {
          "id": {
            "class": "SoftmaxActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000001f"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000001e",
              "bias": "[0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-52630000001d",
                "weights": "[ [ 0.02161199454033596,-0.05006336570006144 ],[ -0.24400763548472423,-0.055548360446179015 ] ]"
              },
              "prev0": {
                "target": "[e8069e1f-07fb-468e-aac0-5d9f79db1c93]"
              }
            }
          }
        }
      ],
      "root": {
        "id": {
          "class": "SoftmaxActivationLayer",
          "id": "bf91e8ca-00fc-4e91-bc12-52630000001f"
        },
        "prev0": {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-52630000001e",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-52630000001d",
              "weights": "[ [ 0.02161199454033596,-0.05006336570006144 ],[ -0.24400763548472423,-0.055548360446179015 ] ]"
            },
            "prev0": {
              "target": "[e8069e1f-07fb-468e-aac0-5d9f79db1c93]"
            }
          }
        }
      }
    }
```



Code from [MindsEyeDemo.scala:214](../../src/test/scala/MindsEyeDemo.scala#L214) executed in 0.00 seconds: 
```java
    val trainingNetwork: SupervisedNetwork = new SupervisedNetwork(model, new EntropyLossLayer)
    val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setTimeout(10, TimeUnit.SECONDS)
    trainer.setTerminateThreshold(0.0)
    trainer
```

Returns: 

```
    com.simiacryptus.mindseye.opt.IterativeTrainer@40ec88ef
```



Code from [MindsEyeDemo.scala:222](../../src/test/scala/MindsEyeDemo.scala#L222) executed in 10.00 seconds: 
```java
    trainer.run()
```

Returns: 

```
    NaN
```



Code from [MindsEyeDemo.scala:246](../../src/test/scala/MindsEyeDemo.scala#L246) executed in 0.09 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.4.png)



Code from [MindsEyeDemo.scala:263](../../src/test/scala/MindsEyeDemo.scala#L263) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (47.0,Map(0 -> 23.80952380952381, 1 -> 63.793103448275865))
```



Code from [MindsEyeDemo.scala:328](../../src/test/scala/MindsEyeDemo.scala#L328) executed in 0.01 seconds: 
```java
    var model: PipelineNetwork = new PipelineNetwork
    val middleSize = Array[Int](15)
    model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model.add(new BiasLayer(middleSize: _*))
    model.add(new AbsActivationLayer())
    model.add(new DenseSynapseLayerJBLAS(middleSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
      override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
    }))
    model.add(new BiasLayer(outputSize: _*))
    model.add(new SoftmaxActivationLayer)
    model
```

Returns: 

```
    {
      "class": "PipelineNetwork",
      "id": "bf91e8ca-00fc-4e91-bc12-526300000022",
      "nodes": [
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
            "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
          },
          "prev0": {
            "target": "[4a477bf5-31ce-4ebb-bef8-e6251d3c7d29]"
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000024",
            "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
              "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
            },
            "prev0": {
              "target": "[4a477bf5-31ce-4ebb-bef8-e6251d3c7d29]"
            }
          }
        },
        {
          "id": {
            "class": "AbsActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000025"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000024",
              "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
                "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
              },
              "prev0": {
                "target": "[4a477bf5-31ce-4ebb-bef8-e6251d3c7d29]"
              }
            }
          }
        },
        {
          "id": {
            "class": "DenseSynapseLayerJBLAS",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000026",
            "weights": "[ [ 0.9005996485518326,-1.3414285988284422 ],[ 2.1656422093308096,-0.9784251699018189 ],[ 0.5756835167414042,2.0942165213801 ],[ -0.38559029174415627,0.6398539092575672 ],[ 0.23163664859553987,-0.22510107938272775 ],[ 0.6910620101329208,-0.3589896108206934 ],[ -0.4263774063917185,-0.7341155294962441 ],[ 0.879073644217262,0.6618101521997762 ],... ]"
          },
          "prev0": {
            "id": {
              "class": "AbsActivationLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000025"
            },
            "prev0": {
              "id": {
                "class": "BiasLayer",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000024",
                "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
              },
              "prev0": {
                "id": {
                  "class": "DenseSynapseLayerJBLAS",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
                  "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
                },
                "prev0": {
                  "target": "[4a477bf5-31ce-4ebb-bef8-e6251d3c7d29]"
                }
              }
            }
          }
        },
        {
          "id": {
            "class": "BiasLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000027",
            "bias": "[0.0, 0.0]"
          },
          "prev0": {
            "id": {
              "class": "DenseSynapseLayerJBLAS",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000026",
              "weights": "[ [ 0.9005996485518326,-1.3414285988284422 ],[ 2.1656422093308096,-0.9784251699018189 ],[ 0.5756835167414042,2.0942165213801 ],[ -0.38559029174415627,0.6398539092575672 ],[ 0.23163664859553987,-0.22510107938272775 ],[ 0.6910620101329208,-0.3589896108206934 ],[ -0.4263774063917185,-0.7341155294962441 ],[ 0.879073644217262,0.6618101521997762 ],... ]"
            },
            "prev0": {
              "id": {
                "class": "AbsActivationLayer",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000025"
              },
              "prev0": {
                "id": {
                  "class": "BiasLayer",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000024",
                  "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                },
                "prev0": {
                  "id": {
                    "class": "DenseSynapseLayerJBLAS",
                    "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
                    "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
                  },
                  "prev0": {
                    "target": "[4a477bf5-31ce-4ebb-bef8-e6251d3c7d29]"
                  }
                }
              }
            }
          }
        },
        {
          "id": {
            "class": "SoftmaxActivationLayer",
            "id": "bf91e8ca-00fc-4e91-bc12-526300000028"
          },
          "prev0": {
            "id": {
              "class": "BiasLayer",
              "id": "bf91e8ca-00fc-4e91-bc12-526300000027",
              "bias": "[0.0, 0.0]"
            },
            "prev0": {
              "id": {
                "class": "DenseSynapseLayerJBLAS",
                "id": "bf91e8ca-00fc-4e91-bc12-526300000026",
                "weights": "[ [ 0.9005996485518326,-1.3414285988284422 ],[ 2.1656422093308096,-0.9784251699018189 ],[ 0.5756835167414042,2.0942165213801 ],[ -0.38559029174415627,0.6398539092575672 ],[ 0.23163664859553987,-0.22510107938272775 ],[ 0.6910620101329208,-0.3589896108206934 ],[ -0.4263774063917185,-0.7341155294962441 ],[ 0.879073644217262,0.6618101521997762 ],... ]"
              },
              "prev0": {
                "id": {
                  "class": "AbsActivationLayer",
                  "id": "bf91e8ca-00fc-4e91-bc12-526300000025"
                },
                "prev0": {
                  "id": {
                    "class": "BiasLayer",
                    "id": "bf91e8ca-00fc-4e91-bc12-526300000024",
                    "bias": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
                  },
                  "prev0": {
                    "id": {
                      "class": "DenseSynapseLayerJBLAS",
                      "id": "bf91e8ca-00fc-4e91-bc12-526300000023",
                      "weights": "[ [ 0.5348930626062534,0.8791177089733158,-0.3670876870430905,0.8321082893530115,-1.257889287992193,-0.5859398095124264,-0.2809521762910377,-1.3407494126965587,... ],[ -0.3749034476338172,-0.0609323147785642,0.0572810878141887,0.0732304736448543,-0.3311987841775831,-1.5554152518468543,-0.920078342882825,-0.0894018998686719,... ] ]"
                    },
                    "prev0... and 2269 more bytes
```



Code from [MindsEyeDemo.scala:214](../../src/test/scala/MindsEyeDemo.scala#L214) executed in 0.00 seconds: 
```java
    val trainingNetwork: SupervisedNetwork = new SupervisedNetwork(model, new EntropyLossLayer)
    val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setTimeout(10, TimeUnit.SECONDS)
    trainer.setTerminateThreshold(0.0)
    trainer
```

Returns: 

```
    com.simiacryptus.mindseye.opt.IterativeTrainer@507ca1ed
```



Code from [MindsEyeDemo.scala:222](../../src/test/scala/MindsEyeDemo.scala#L222) executed in 10.00 seconds: 
```java
    trainer.run()
```

Returns: 

```
    NaN
```



Code from [MindsEyeDemo.scala:246](../../src/test/scala/MindsEyeDemo.scala#L246) executed in 0.08 seconds: 
```java
    plotXY(gfx)
```

Returns: 

![Result](2d_simple.5.png)



Code from [MindsEyeDemo.scala:263](../../src/test/scala/MindsEyeDemo.scala#L263) executed in 0.00 seconds: 
```java
    overall → byCategory
```

Returns: 

```
    (87.0,Map(0 -> 92.3076923076923, 1 -> 83.60655737704919))
```



