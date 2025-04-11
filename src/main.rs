use ndarray::array;
mod fully_connected_network; 
use fully_connected_network::{LinearReLU, LossFunction, FullyConnectedNet};
use ndarray::{Array1, Array2};

fn main() {
    // 1. 测试 LinearReLU
    let input = array![[1.0, -2.0, 3.0], [0.5, -0.5, 1.5]]; // 输入数据
    let weights = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]; // 权重
    let bias = array![0.1, 0.2]; // 偏置

    let (output, cache) = LinearReLU::forward(&input, &weights, &bias);
    println!("Linear_ReLU 前向传播输出:\n{}", output);

    let dout = array![[1.0, 1.0], [1.0, 1.0]];

    let (dx, dw, db) = LinearReLU::backward(&dout, cache);
    println!("Linear_ReLU 反向传播梯度 dx:\n{}", dx);
    println!("Linear_ReLU 反向传播梯度 dw:\n{}", dw);
    println!("Linear_ReLU 反向传播梯度 db:\n{}", db);

    // 2. 测试完整网络
    let input_dim = 4;  // 输入层的维度
    let hidden_dims = vec![5, 3]; // 隐藏层的维度
    let num_classes = 3; // 输出层的维度
    let reg = 0.1; // 正则化系数
    let loss_fn = LossFunction::Softmax; // 损失函数选择 Softmax

    let net = FullyConnectedNet::new(hidden_dims, input_dim, num_classes, reg, loss_fn);

    // 模型数据X: (5 * 4)
    let X = Array2::<f64>::from_shape_vec(
        (5, 4),
        vec![
            1.0, 0.5, -1.5, 0.3,   // 样本 1
            0.3, -0.2, 1.0, -0.1,   // 样本 2
            0.7, 0.9, 0.4, 0.8,     // 样本 3
            -1.2, 0.5, 1.2, -0.7,   // 样本 4
            0.6, -0.8, -0.5, 1.4    // 样本 5
        ]
    ).unwrap();

    // 目标标签y: (5 * 1) 对应标签是 0、1 或 2，表示类别
    let y = Array1::<usize>::from_vec(vec![0, 1, 2, 0, 1]);

    // 3. 执行前向传播
    let (scores, _caches) = net.forward(&X);
    println!("Network output (scores):\n{}", scores);


    let (loss, grads_w, grads_b) = net.loss(&X, &y);
    println!("Loss: {}", loss);
    println!("Gradients for weights (W):\n{:?}", grads_w);
    println!("Gradients for biases (b):\n{:?}", grads_b);
}
