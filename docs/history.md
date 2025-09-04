# Simple Online and Realtime Tracking history

Since the publication of the **Simple Online and Realtime Tracking (SORT)** algorithm in 2016, the field of multi-object tracking (MOT) has seen significant innovations, largely building upon SORT's foundational "tracking-by-detection" paradigm. The primary advancements have focused on improving the data association step, making trackers more robust to occlusions, similar-looking objects, and unpredictable motion.

***

### Key Innovations and Algorithms

#### 1. DeepSORT: Integrating Appearance Features
SORT's main limitation was its reliance solely on motion and bounding box overlap (Intersection over Union, or IoU) for data association. This made it prone to "ID-switches" when objects moved erratically or passed each other closely.

In 2017, the **DeepSORT** algorithm addressed this by introducing a **deep association metric**. It uses a pre-trained deep convolutional neural network (CNN) to extract a feature vector representing the object's appearance. This feature vector, combined with the motion-based Kalman filter, provides a much more robust and accurate way to match detections to existing tracks, significantly reducing ID-switches. DeepSORT's success demonstrated the critical importance of using both motion and appearance information for effective tracking.



#### 2. ByteTrack: Considering Low-Confidence Detections
Traditional trackers, including SORT and DeepSORT, often discard low-confidence detections from the object detector to reduce noise and false positives. However, this practice can lead to lost tracks, especially when objects are partially occluded or far away.

**ByteTrack** (2021) introduced a novel "Byte" association scheme. Instead of discarding low-confidence detections, it attempts to re-associate them with existing tracks. It first matches high-confidence detections and then uses the unmatched, low-confidence detections to recover tracks that might have been lost. This "Byte" strategy significantly improves tracking performance in crowded scenes and under occlusions by preventing premature track termination.

#### 3. OC-SORT: Rethinking the Kalman Filter
The Kalman filter, a core component of SORT, assumes that object motion is linear. This assumption can be a major weakness in real-world scenarios where objects move non-linearly.

**Observation-Centric SORT (OC-SORT)** (2022) challenged this estimation-centric approach. Instead of solely relying on the Kalman filter's prediction, OC-SORT uses the object observations (detections) themselves to correct for accumulated errors during occlusions. This approach, which can be thought of as a "virtual trajectory" over the occlusion period, allows for more robust tracking of non-linear movements while maintaining the simplicity and speed of the original SORT framework.

#### 4. BoT-SORT: A "Bag of Tricks"
**BoT-SORT** (2022) builds upon the success of ByteTrack by incorporating a "bag of tricks" to improve tracking accuracy. It introduces several key improvements:

* **Improved Bounding Box Prediction:** It refines the Kalman filter to directly estimate the object's width and height, not just its aspect ratio, leading to more accurate bounding boxes.
* **Camera Motion Compensation:** It includes a mechanism to compensate for camera movement, which is crucial for applications like autonomous vehicles or drone footage.
* **Enhanced Data Association:** It uses a more sophisticated approach for combining motion and appearance cues during the matching process, leading to better results.

***

### State-of-the-Art Applications
The innovations in object tracking have enabled a wide range of sophisticated applications:

* **Autonomous Driving:** Tracking pedestrians, other vehicles, and cyclists in real-time is fundamental for safe navigation and decision-making. Algorithms like BoT-SORT are highly relevant here due to their ability to handle camera motion and crowded scenes.
* **Security and Surveillance:** Tracking individuals or objects across multiple camera views for threat detection and anomaly analysis. The reduced ID-switches from algorithms like DeepSORT are critical for maintaining a consistent identity.
* **Sports Analytics:** Analyzing player movements, ball trajectories, and team strategies. The ability to track similar-looking athletes in a fast-paced environment, addressed by methods like Deep HM-SORT, is essential.
* **Robotics:** Enabling robots to interact with their environment by accurately tracking objects for manipulation, navigation, and human-robot collaboration.
* **Retail Analytics:** Monitoring customer foot traffic, dwell times, and product interactions to optimize store layouts and marketing strategies.