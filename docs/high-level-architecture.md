\# High-Level Architecture of Object Trackers



A variety of architectures are used for object tracking and modeling with multiple sensors. Still, they generally fall into three main categories based on where and when the sensor data is fused. These architectures are \*\*Centralized\*\*, \*\*Decentralized\*\* (or distributed), and \*\*Hybrid\*\*.



---



\### Centralized Architecture



In a centralized architecture, data from all sensors is sent to a single processing unit. This raw sensor data is then fused and processed to create a single, unified model of the environment.



\* \*\*How it works:\*\* This approach processes the raw data streams from all sensors (e.g., point clouds from LiDAR, pixel data from cameras, raw radar detections) at a central location. It then applies a single tracking algorithm, often a Bayesian filter like a \*\*Kalman Filter\*\* or a \*\*Particle Filter\*\*, to estimate the object's state (position, velocity, acceleration). The model is then reasoned with to confirm the object detection. 

\* \*\*Advantages:\*\* This method can be very accurate because it uses all available information from every sensor at the earliest possible stage. The correlations and nuances between different sensor types can be fully exploited, leading to a more robust and complete model.

\* \*\*Disadvantages:\*\* It's computationally expensive and requires a high-bandwidth communication system to transmit all the raw data. The central processing unit becomes a single point of failure.



---



\### Decentralized Architecture



A decentralized architecture processes data locally at each sensor or a dedicated sensor processing unit. Each sensor unit creates its local model of the environment, and then these local models (not the raw data) are fused.



\* \*\*How it works:\*\* Each sensor's data is independently processed to detect objects and create a local track (a model of an object's state). For example, a camera might detect an object's position and size, while a radar might detect its range and velocity. These individual tracks are then sent to a central unit, where a fusion algorithm combines them.

\* \*\*Advantages:\*\* This approach reduces communication overhead because only the processed tracks, not the raw data, are transmitted. It's also more robust to single-sensor failures, as the system can continue to operate with the remaining sensors. It's also highly scalable.

\* \*\*Disadvantages:\*\* Since the raw data isn't shared, it's harder to exploit the complementary strengths of different sensors fully. The final fused track may not be as accurate as one from a centralized system, and there can be complexities in aligning the different local models.



---



\### Hybrid Architecture



A hybrid architecture combines elements of both centralized and decentralized approaches to balance their advantages and disadvantages.



\* \*\*How it works:\*\* This is the most common approach in many real-world applications. A standard hybrid model is to use \*\*Centralized-level fusion\*\* for some sensors and \*\*Decentralized-level fusion\*\* for others. For instance, a system might fuse raw camera and LiDAR data to achieve highly accurate object detection, then fuse that result with a radar-generated track (which is already a processed, local model) to get a complete picture. 

\* \*\*Advantages:\*\* It offers an outstanding balance between performance and computational cost. You can choose to fuse raw data from highly complementary sensors, while using a decentralized approach for sensors that are more distinct or less critical.

\* \*\*Disadvantages:\*\* The complexity of the architecture increases, as you need to manage multiple fusion points and algorithms. It requires careful design to ensure the different fusion methods work together seamlessly.

