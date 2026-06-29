use sodals_industrial::introspection::*;

#[test]
fn test_sensor_calibration() {
    let mut template = [0.0; TEMPLATE_SIZE];
    template[0..5].copy_from_slice(&[20.0, 20.5, 21.0, 20.8, 20.2]);
    let calib = SensorCalibration::new(1, 20.5, 0.5, template, 101);
    
    // Normal values shouldn't trigger anomaly (3-sigma with deviation=3.0)
    assert!(!calib.is_anomalous(21.0, 3.0));
    
    // Anomaly values should trigger anomaly
    assert!(calib.is_anomalous(23.0, 3.0));
}

#[test]
fn test_wronskian_computation() {
    // Test linear dependence (should be near 0)
    // f1 = sin(t), f2 = cos(t) -> W(t) = sin(t)*(-sin(t)) - cos(t)*(cos(t)) = -1
    // Let's test with simple discrete arrays
    let dt = 0.1;
    // f1 = [sin(0), sin(0.1)] = [0.0, 0.099833]
    // f2 = [cos(0), cos(0.1)] = [1.0, 0.995004]
    let f1 = [0.0, 0.099833];
    let f2 = [1.0, 0.995004];
    
    let w = compute_wronskian(&f1, &f2, dt);
    assert!((w - (-1.0)).abs() < 0.05);
}

#[test]
fn test_jacobian_and_rul() {
    let prev = StateSpace::new(50.0, 2.0, 1500.0, 2.5);
    let curr = StateSpace::new(52.0, 2.1, 1505.0, 2.7);
    let dt = 0.5;
    
    let jacobian = compute_jacobian(&curr, &prev, dt);
    
    // Check main diagonal: dy/dx / dt
    // jac[0][0] = (52.0 - 50.0) / (52.0 - 50.0) / 0.5 = 1.0 / 0.5 = 2.0
    assert!((jacobian[0][0] - 2.0).abs() < 1e-5);
    
    // Compute dominant eigenvalue
    let lambda = power_iteration_4x4(&jacobian, 50, 1e-6);
    assert!(lambda > 0.0);
    
    // Compute RUL
    let rul = estimate_rul(&jacobian, 100.0, 1e-5);
    assert!(rul > 0.0);
}

#[test]
fn test_lsq_update() {
    // 3 samples (N=3), 2 sensors (SENSORS=2)
    let a = [
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 1.0],
    ];
    let b = [5.0, 4.0, 3.0];
    let w_old = [0.5, 0.5];
    let learning_rate = 0.05;
    let mut w_new = [0.0; 2];
    
    update_weights_lsq(&a, &b, &w_old, learning_rate, &mut w_new);
    
    // Prediction for 1st sample: 1*0.5 + 2*0.5 = 1.5, error = 5.0 - 1.5 = 3.5
    // Prediction for 2nd sample: 2*0.5 + 1*0.5 = 1.5, error = 4.0 - 1.5 = 2.5
    // Prediction for 3rd sample: 1*0.5 + 1*0.5 = 1.0, error = 3.0 - 1.0 = 2.0
    // Residuals = [3.5, 2.5, 2.0]
    // Gradient[0] = a[0][0]*res[0] + a[1][0]*res[1] + a[2][0]*res[2] = 1*3.5 + 2*2.5 + 1*2.0 = 10.5
    // Gradient[1] = a[0][1]*res[0] + a[1][1]*res[1] + a[2][1]*res[2] = 2*3.5 + 1*2.5 + 1*2.0 = 11.5
    // w_new[0] = 0.5 + 0.05 * 10.5 = 1.025
    // w_new[1] = 0.5 + 0.05 * 11.5 = 1.075
    
    assert!((w_new[0] - 1.025).abs() < 1e-5);
    assert!((w_new[1] - 1.075).abs() < 1e-5);
}

#[test]
fn test_top_features() {
    let scores = [0.1, 0.9, 0.4, 0.2, 0.8, 0.3];
    let selected: [usize; 3] = select_top_features(&scores);
    
    // Top 3 should be 1 (0.9), 4 (0.8), and 2 (0.4)
    assert_eq!(selected[0], 1);
    assert_eq!(selected[1], 4);
    assert_eq!(selected[2], 2);
}

#[test]
fn test_newton_interpolation() {
    // Interpolate points on y = x^2
    let x = [1.0, 2.0, 3.0];
    let y = [1.0, 4.0, 9.0];
    
    let val = newton_interpolate(&x, &y, 2.5);
    // 2.5^2 = 6.25
    assert!((val - 6.25).abs() < 1e-5);
}

#[test]
fn test_thermodynamic_grammar() {
    // Valid case: Winding is warmer than ambient, cooling is positive
    assert!(validate_thermodynamic_grammar(80.0, 25.0, 2.5));
    
    // Invalid: negative flow
    assert!(!validate_thermodynamic_grammar(80.0, 25.0, -1.0));
    
    // Invalid: winding colder than ambient
    assert!(!validate_thermodynamic_grammar(15.0, 25.0, 2.5));
}
