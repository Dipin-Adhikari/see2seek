# See2Seek: DINOv2-Based Zero-Shot Object Navigation

See2Seek investigates the use of **DINOv2** as the visual observation encoder for **Zero-Shot Object Navigation (ObjectNav)** in the **RoboTHOR** simulation environment.

The project builds upon the **ZSON (Zero-Shot Object Navigation)** framework by replacing CLIP's image encoder with **DINOv2 ViT-B/14** while retaining CLIP's text encoder for goal representation. A **PPO-based recurrent navigation policy** is trained to navigate toward target objects specified by natural language without requiring task-specific visual supervision.

The primary objective is to evaluate whether DINOv2's richer spatial representations improve navigation performance over the original CLIP-based approach using standard ObjectNav metrics such as **Success Rate (SR)** and **Success weighted by Path Length (SPL)**.