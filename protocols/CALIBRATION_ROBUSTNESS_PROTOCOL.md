# Frozen Calibration Robustness Protocol

FedCANTO and its exact Router-only control are compared on compound CIFAR-10,
CIFAR-100, and Office-Home using paired seeds 40--44. The held-out server
calibration fraction is varied over 1%, 2%, 4%, and 8%. A separate 4% condition
draws 80% of calibration examples from the lower half of labels. Test labels are
logged only for diagnostics and never affect routing. All candidate spans,
evidence coefficients, momentum, local optimization, and the 0.001 selection
margin remain frozen.

