# Representation Flow for Action Recognition

## Representation Flow Layer
![tsf](/examples/flow-layer.png?raw=true "repflow")

Authors of the base paper introduced the representation flow layer, which can be found in [rep_flow_layer.py](rep_flow_layer.py). This layer iteratively estimates the flow, can be applied to CNN feature maps, and is fully learnable to maximize classification performance.

We perform a comparative analysis on it to assess it's generalizability in Dark Conditions.

## Network
![model overview](/examples/flow-in-network.png?raw=true "model overview")


## Results

|  Method   | HMDB  |  Kinetics-RGB | ARiD |  Runtime   | 
| --------- | ----- | ------------- | ---- | ---------- | 
| 2D RGB    | 53.4  |      61.3     | 37.2 |  225±15ms  |
| 2 stream  | 62.4  |      64.5     | 43.1  | 8546±147ms |
| Rep Flow  | 73.5  |      68.5     | 41.8 |  524±24ms  |





## Visualization of learned flows
Examples of representation flows for various actions. The representation flow is computed after the 3rd residual block and captures some sematic motion information. At this point, the representations are low dimensional (28x28).


<img src="https://piergiaj.github.io/rep-flow-site/box_flow_c15.gif"> <img src="https://piergiaj.github.io/rep-flow-site/swing_flow_c1.gif"> <img src="https://piergiaj.github.io/rep-flow-site/handstand_flow_c21.gif">


Examples of representation flows for different channels for "clapping." Some channels capture the hand motion, while other channels focus on different features/motion patterns not present in this clip.

<img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c8.gif"> <img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c16.gif"> <img src="https://piergiaj.github.io/rep-flow-site/clap_flow_c21.gif">


## References
        @inproceedings{repflow2019,
              title={Representation Flow for Action Recognition},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              author={AJ Piergiovanni and Michael S. Ryoo},
              year={2019}
        }
