# MUSE Architecture Analysis and Classification

## Executive Summary

The **MUSE (MUlti-sensor State Estimator)** repository from IIT's Dynamic Legged Systems Lab implements a **Hierarchical Time-Scale Separation Architecture** with some **Event-Driven** characteristics. This places it primarily in **Architecture Pattern #1** from our classification, with elements of **Architecture Pattern #3**.

## Repository Overview

- **Repository**: https://github.com/iit-DLSLab/muse
- **Purpose**: Real-time multi-sensor state estimation for quadruped robots
- **Target Platform**: Quadruped robots (tested on Unitree Aliengo and ANYmal B300)
- **Implementation**: C++ with ROS integration
- **Primary Focus**: State estimation (not model predictive control)

## Architectural Classification: **Hierarchical Time-Scale Separation**

### Key Evidence for Hierarchical Architecture:

1. **Clear Layered Structure**:
   - **High-Level Components**: Camera/LiDAR odometry (slower, ~10-30 Hz)
   - **Medium-Level Components**: Attitude observer with nonlinear observer and eXogenous Kalman Filter
   - **Low-Level Components**: IMU integration, joint encoder processing (~1000 Hz)
   - **Fusion Layer**: Kalman Filter for sensor fusion

2. **Multi-Rate Sensor Integration**:
   - **Fast Rate**: IMU data (1000 Hz), joint encoders (1000 Hz)
   - **Medium Rate**: Force/torque sensors (100-500 Hz)
   - **Slow Rate**: Camera odometry (30 Hz), LiDAR odometry (10-20 Hz)

3. **Unidirectional Information Flow**:
   - Higher-level exteroceptive sensors provide reference trajectories
   - Lower-level proprioceptive sensors handle real-time state propagation
   - Clear hierarchy from planning to execution

### Event-Driven Elements (Secondary Classification):

MUSE also incorporates **Event-Driven** characteristics:

1. **Slip Detection Module**: 
   - Dynamically adjusts sensor weights based on detected foot slippage
   - Event-triggered reconfiguration of the Kalman filter parameters

2. **Adaptive Covariance Adjustment**:
   - Contact probability model triggers changes in observation weights
   - Dynamic switching between sensor modalities based on conditions

## Technical Architecture Details

### Core Components:

1. **Attitude Observer (AO)**:
   - Nonlinear Observer (NLO) for orientation estimation
   - eXogenous Kalman Filter (XKF) for bias estimation
   - Operates at IMU frequency (~1000 Hz)

2. **Leg Odometry (LO)**:
   - Joint state processing
   - Robot kinematics/dynamics computation
   - Ground reaction force analysis
   - Medium frequency operation (~100-500 Hz)

3. **Exteroceptive Sensors**:
   - Camera odometry (T265 tracking camera in lab experiments)
   - LiDAR odometry (KISS-ICP algorithm for outdoor experiments)
   - Low frequency operation (~10-30 Hz)

4. **Slip Detection (SD)**:
   - Kinematics-based strategy for concurrent leg slippage detection
   - Event-driven activation when inconsistencies detected

5. **Sensor Fusion (SF)**:
   - Central Kalman Filter for multi-sensor integration
   - Adaptive covariance based on contact probabilities
   - Operates at medium frequency (~100 Hz)

### Multi-Rate Implementation Strategy:

```
High Level (Slow):     Camera/LiDAR Odometry (10-30 Hz)
                              ↓
Medium Level:          Sensor Fusion KF (100 Hz)
                              ↓  
Low Level (Fast):      IMU + Encoders (1000 Hz)
```

## Comparison with Our Four Architecture Patterns

| Aspect | MUSE Implementation | Pattern Match |
|--------|-------------------|---------------|
| **Structure** | Clear hierarchical layers with different frequencies | ✅ Hierarchical |
| **Sensor Integration** | Multi-rate with frequency-appropriate processing | ✅ Hierarchical |
| **Event Handling** | Slip detection triggers adaptive behavior | ⚡ Event-Driven |
| **Real-time Performance** | Good, validated in real experiments | ✅ Hierarchical |
| **Complexity** | Medium (well-structured but substantial) | ✅ Hierarchical |

## Key Architectural Insights

### Strengths of MUSE's Approach:

1. **Clear Separation of Concerns**: Each sensor type processed at appropriate frequency
2. **Modular Design**: ROS plugin architecture allows easy extension
3. **Robust State Estimation**: Combines complementary sensor modalities effectively
4. **Real-World Validation**: Demonstrated on multiple robot platforms

### Multi-Rate Strategy:

1. **Fast Loop (1000 Hz)**: IMU-driven state propagation
2. **Medium Loop (100 Hz)**: Kalman filter fusion with leg odometry
3. **Slow Loop (10-30 Hz)**: Exteroceptive corrections from vision/LiDAR
4. **Event-Driven**: Slip detection and adaptive covariance adjustment

## Relevance to Multi-Rate MPC

While MUSE is primarily a **state estimator** rather than a **controller**, its architecture provides excellent insights for multi-rate MPC design:

### Applicable Design Patterns:

1. **Hierarchical Structure**: Clear time-scale separation works well for multi-rate systems
2. **Plugin Architecture**: ROS-based modular design enables easy extension
3. **Adaptive Behavior**: Event-driven slip detection shows how to handle exceptional conditions
4. **Sensor Fusion**: Multi-rate integration strategies directly applicable to MPC

### Potential MPC Extensions:

The MUSE architecture could be extended for multi-rate MPC by:

1. **Adding Control Layers**: 
   - Strategic MPC (slow, ~1-10 Hz) for trajectory planning
   - Tactical MPC (fast, ~100-1000 Hz) for tracking control

2. **Leveraging Existing Infrastructure**:
   - Use existing sensor fusion for state feedback
   - Extend slip detection for control adaptation
   - Maintain hierarchical time-scale separation

## Conclusion

**MUSE implements a Hierarchical Time-Scale Separation Architecture (#1)** with significant **Event-Driven elements (#3)**. This hybrid approach provides:

- ✅ Clear, maintainable structure
- ✅ Good real-time performance 
- ✅ Robust multi-sensor integration
- ✅ Adaptive behavior for exceptional conditions

The architecture demonstrates that hierarchical approaches can be highly effective for real-world robotic systems, especially when augmented with event-driven adaptive mechanisms. This validates our analysis that most practical systems use hybrid architectures combining multiple patterns.

For your multi-rate MPC prototype, MUSE's architecture provides an excellent reference implementation showing how to structure multi-rate systems with clear time-scale separation while maintaining real-time performance and robustness.