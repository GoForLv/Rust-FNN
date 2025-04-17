use ndarray::{array, range};
mod fully_connected_network; 
use fully_connected_network::{LinearReLU, LossFunction, FullyConnectedNet};
use ndarray::{Array1, Array2};

use csv::Reader;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;

fn load_iris_data() -> ((usize, usize), usize, usize, Vec<f64>, Vec<usize>) {

    let mut rdr = Reader::from_path("src\\iris.csv").unwrap();
    let mut features: Vec<f64> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();

    // println!("{:?}", rdr.headers());
    let mut n_features = 0;

    for record in rdr.records() {
        let record = record.unwrap();
        n_features = record.len() - 1;
        for i in 0..n_features {
            let data: f64 = record[i].parse().unwrap();
            features.push(data);
        }

        let label = match &record[n_features] {
            "setosa"     => 0,
            "versicolor" => 1,
            "virginica"  => 2,
            _ => 3,
        };
        labels.push(label);
    }

    let features_size = (features.len() / n_features, n_features);
    let n_classes = 3;

    (features_size, labels.len(), n_classes, features, labels)
}

fn main() {
    let (features_size, labels_size, n_classes, features, labels) = load_iris_data();

    // 模型数据: (150, 4)
    let features = Array2::<f64>::from_shape_vec(features_size, features).unwrap();

    // 目标标签: 150 (1, 2, 3)
    let labels = Array1::<usize>::from_vec(labels);

    // println!("{:?} \n {:?}",features, labels);
    println!("features_size: {:?}, labels_size: {:?}, n_classes: {:?}", features_size, labels_size, n_classes);

    let input_dim = features_size.1;  // 输入层的维度
    let hidden_dims = vec![16, 8]; // 隐藏层的维度
    let num_classes = n_classes; // 输出层的维度
    let reg = 0.1; // 正则化系数
    let loss_fn = LossFunction::Softmax; // 损失函数选择 Softmax

    let mut net = FullyConnectedNet::new(hidden_dims, input_dim, num_classes, reg, loss_fn);

    let n_epochs = 1000;
    let lr = 0.0005;

    for epoch in 0..n_epochs {
        let (loss, grads_w, grads_b) = net.loss(&features, &labels);
        for (wi, grad_w) in &grads_w {
            net.weight_params.insert(wi.to_string(), net.weight_params[wi].clone() - lr * grad_w);
        }

        for (bi, grad_b) in &grads_b {
            net.bias_params.insert(bi.to_string(), net.bias_params[bi].clone() - lr * grad_b);
        }

        if epoch % 50 == 0 {
            println!("Epoch: {}, Loss: {}", epoch, loss);
        }
    }

    let (scores, _caches) = net.forward(&features);
    let mut pred : Vec<usize> = Vec::new();
    for i in 0..scores.dim().0 {
        let mut max_idx = 0;
        let mut max_val = -1e9;
        for j in 0..scores.dim().1 {
            if scores[[i, j]] > max_val {
                max_val = scores[[i, j]];
                max_idx = j;
            }
            pred.push(max_idx);
        }
    }
    println!("{:?}", pred);

    let mut accuracy : f64 = 0.0;
    for i in 0..labels.len() {
        if labels[i] == pred[i] {
            accuracy += 1.0;
        }
    }
    println!("accuracy: {}, accuracy_rate: {}.", accuracy, 100.0 * accuracy / labels.len().to_f64().unwrap());
}
