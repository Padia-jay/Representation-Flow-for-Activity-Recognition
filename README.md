# Representation Flow for Action Recognition

## Representation Flow Layer
![tsf](/examples/flow-layer.png?raw=true "repflow")

Authors of the base paper introduced the representation flow layer, which can be found in [rep_flow_layer.py](rep_flow_layer.py). This layer iteratively estimates the flow, can be applied to CNN feature maps, and is fully learnable to maximize classification performance.

We perform a comparative analysis on it to assess it's generalizability in Dark Conditions.

## Network
![model overview](/examples/flow-in-network.png?raw=true "model overview")


## Results

<!-- |  Method | Kinetics-400  |  HMDB | Runtime | 
| ------------- | ------------- | ----------- | ------- | 
| 2D Two-Stream | 64.5  | 66.6  | 8546ms  |
| TVNet (+RGB)  | -     | 71.0  | 785ms |
| (2+1)D Two-Stream | 75.4 | 78.7 | 8623ms |
| I3D Two-stream | 74.2 | 80.7 | 9354ms |
| (2+1)D + Rep-Flow | 75.5 | 77.1 | 622ms |
| (2+1)D + Flow-of-flow | 77.1 | 81.1 | 654ms | -->





# Visualization of learned flows
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
