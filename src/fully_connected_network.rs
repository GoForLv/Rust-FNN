use ndarray::{Array2, Array1, Axis};
use ndarray::Zip;
use std::collections::HashMap;
use ndarray_rand::RandomExt;

pub struct Linear;

impl Linear {
    pub fn forward(x: &Array2<f64>, w: &Array2<f64>, b: &Array1<f64>) -> (Array2<f64>, (Array2<f64>, Array2<f64>, Array1<f64>)) {
        let a = x.dot(w) + b;
        let cache = (x.clone(), w.clone(), b.clone());
        (a, cache)
    }

    pub fn backward(dout: &Array2<f64>, cache: (Array2<f64>, Array2<f64>, Array1<f64>)) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let (x, w, _b) = cache;
        let dx = dout.dot(&w.t());
        let dw = x.t().dot(dout);
        let db = dout.sum_axis(Axis(0));
        (dx, dw, db)
    }
}

pub struct ReLU;

impl ReLU {
    pub fn forward(x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let out = x.mapv(|v| v.max(0.0));
        (out, x.clone())
    }

    pub fn backward(dout: &Array2<f64>, cache: &Array2<f64>) -> Array2<f64> {
        let mut dx = dout.clone();
        let x = cache;
        for (i, elem) in dx.indexed_iter_mut() {
            if x[i] <= 0.0 {
                *elem = 0.0; 
            } 
        }
        dx
    }
}

pub enum Cache {
    LinearCache {
        X: Array2<f64>,  // 输入
        W: Array2<f64>,  // 权重
        b: Array1<f64>,  // 偏置
    },
    ReLUCache {
        X: Array2<f64>,  // 输入
    },
    LinearReLUCache {
        LinearCache: (Array2<f64>, Array2<f64>, Array1<f64>),  // Linear 层的缓存
        ReLUCache: Array2<f64>,  // ReLU 层的缓存
    },
}
pub struct LinearReLU;

impl LinearReLU {
    pub fn forward(x: &Array2<f64>, w: &Array2<f64>, b: &Array1<f64>) -> (Array2<f64>, Cache) {
        // 先执行 Linear 层的前向传播
        let (a, fc_cache) = Linear::forward(x, w, b);

        // 然后执行 ReLU 层的前向传播
        let (out, relu_cache) = ReLU::forward(&a);

        // 将 Linear 和 ReLU 层的缓存包装在一个 `Cache::LinearReLUCache` 中
        let cache = Cache::LinearReLUCache {
            LinearCache: fc_cache,
            ReLUCache: relu_cache,
        };

        (out, cache)
    }

    pub fn backward(dout: &Array2<f64>, cache: Cache) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        // 解包 `cache`，获取 Linear 和 ReLU 层的缓存
        if let Cache::LinearReLUCache {
            LinearCache: fc_cache,
            ReLUCache: relu_cache,
        } = cache {
            // 先进行 ReLU 层的反向传播
            let da = ReLU::backward(dout, &relu_cache);

            // 然后进行 Linear 层的反向传播
            let (dx, dw, db) = Linear::backward(&da, fc_cache);

            (dx, dw, db)
        } else {
            panic!("Cache type mismatch during backward pass.");
        }
    }
}


fn svm_loss(x: &Array2<f64>, y: &Array1<usize>) -> (f64, Array2<f64>) {
    let N = x.shape()[0];
    let C = x.shape()[1];

    let mut correct_class_scores = Array1::<f64>::zeros(N);
    for (i, label) in y.iter().enumerate() {
        correct_class_scores[i] = x[(i, *label)];
    }

    let correct_class_scores_broadcasted = Array2::from_shape_fn((N, C), |(i, _)| correct_class_scores[i]);

    let mut margins = x.clone();
    Zip::from(&mut margins)
        .and(&correct_class_scores_broadcasted)
        .for_each(|m, c| *m = (*m - c + 1.0).max(0.0));

    for (i, label) in y.iter().enumerate() {
        margins[(i, *label)] = 0.0;
    }

    let loss = margins.sum() / N as f64;

    let mut dx = Array2::<f64>::zeros((N, C));
    let mut num_pos = Array1::<usize>::zeros(N);
    for i in 0..N {
        for j in 0..C {
            if margins[(i, j)] > 0.0 {
                dx[(i, j)] = 1.0;
                num_pos[i] += 1;
            }
        }
    }

    for i in 0..N {
        dx[(i, y[i])] -= num_pos[i] as f64;
    }

    dx /= N as f64;

    (loss, dx)
}

fn softmax_loss(x: &Array2<f64>, y: &Array1<usize>) -> (f64, Array2<f64>) {
    let N = x.shape()[0];
    let C = x.shape()[1];

    let row_max = x.fold_axis(Axis(1), f64::NEG_INFINITY, |&a, &b| a.max(b));
    
    let shifted_logits = x - &row_max.into_shape_with_order((N, 1)).unwrap().broadcast((N, C)).unwrap();

    let Z = shifted_logits.mapv(|x| x.exp()).sum_axis(Axis(1));

    let log_probs = shifted_logits - &Z.into_shape_with_order((N, 1)).unwrap().broadcast((N, C)).unwrap().mapv(f64::ln);

    let probs = log_probs.mapv(f64::exp);

    let mut loss = 0.0;
    for i in 0..N {
        loss -= log_probs[(i, y[i])];
    }
    loss /= N as f64;

    let mut dx = probs.clone();
    for i in 0..N {
        dx[(i, y[i])] -= 1.0;
    }

    dx /= N as f64;

    (loss, dx)
}


pub struct FullyConnectedNet {
    num_layers: usize,
    weight_params: HashMap<String, Array2<f64>>, // 存储权重 W
    bias_params: HashMap<String, Array1<f64>>,   // 存储偏置 b
    reg: f64,
    loss_fn: LossFunction,  // 用于选择损失函数
}

pub enum LossFunction {
    SVM,
    Softmax,
}

impl FullyConnectedNet {
    pub fn new(
        hidden_dims: Vec<usize>,
        input_dim: usize,
        num_classes: usize,
        reg: f64,
        loss_fn: LossFunction,  // 传入损失函数类型
    ) -> Self {
        let mut layer_dims = vec![input_dim];
        layer_dims.extend(hidden_dims);
        layer_dims.push(num_classes);
        
        let mut weight_params = HashMap::new();
        let mut bias_params = HashMap::new();

        // 初始化权重和偏置
        for i in 0..layer_dims.len() - 1 {
            let W = Array2::<f64>::random((layer_dims[i], layer_dims[i + 1]), ndarray_rand::rand_distr::StandardNormal) * 1.0;
            let b = Array1::<f64>::zeros(layer_dims[i + 1]);

            weight_params.insert(format!("W{}", i + 1), W);
            bias_params.insert(format!("b{}", i + 1), b);
        }

        FullyConnectedNet { 
            num_layers: layer_dims.len(), 
            weight_params, 
            bias_params, 
            reg, 
            loss_fn, 
        }
    }

    pub fn forward(&self, X: &Array2<f64>) -> (Array2<f64>, Vec<Cache>) {
        let mut current_input = X.clone();
        let mut caches = Vec::new();
    
        // 通过每一层的前向传播
        for i in 1..self.num_layers - 1 {
            // 使用借用而非移动
            let W = &self.weight_params[&format!("W{}", i)];
            let b = &self.bias_params[&format!("b{}", i)];
    
            let (new_input, cache) = LinearReLU::forward(&current_input, W, b);
            current_input = new_input;
            caches.push(cache); // 存储 LinearReLU 层的 cache
        }
    
        // 最后一层是线性层
        let W_last = &self.weight_params[&format!("W{}", self.num_layers - 1)];
        let b_last = &self.bias_params[&format!("b{}", self.num_layers - 1)];
        let (scores, _cache) = Linear::forward(&current_input, W_last, b_last);
        caches.push(Cache::LinearCache {
            X: current_input.clone(),
            W: W_last.clone(),
            b: b_last.clone(),
        });
    
        (scores, caches)
    }
    
    

    pub fn backward(&self, dout: &Array2<f64>, caches: Vec<Cache>) -> (HashMap<String, Array2<f64>>, HashMap<String, Array1<f64>>) {
        let mut grads_w = HashMap::new(); // 存储 dW
        let mut grads_b = HashMap::new(); // 存储 db
        let mut dh = dout.clone();
    
        // 从最后一层开始反向传播
        for i in (1..self.num_layers).rev() {
            let (dout, mut dW, db);
            match &caches[i - 1] {
                Cache::LinearCache { X, W, b } => {
                    // 解包 LinearCache
                    (dout, dW, db) = Linear::backward(&dh, (X.clone(), W.clone(), b.clone()));
                }
                Cache::LinearReLUCache {
                    LinearCache: (X, W, b),
                    ReLUCache,
                } => {
                    // 解包 LinearReLUCache
                    (dout, dW, db) = LinearReLU::backward(&dh, Cache::LinearReLUCache {
                        LinearCache: (X.clone(), W.clone(), b.clone()),
                        ReLUCache: ReLUCache.clone(),
                    });
                }
                Cache::ReLUCache { .. } => {
                    // 如果你不处理 ReLUCache，可以在这里添加一个 `todo!()`，或者根据需求处理它
                    todo!("Handle ReLUCache case if needed");
                }
            }
    
            // 加入 L2 正则化的梯度
            let W = &self.weight_params[&format!("W{}", i)];

            dW = dW + W * (2.0 * self.reg);  // 使用 add_assign 来进行矩阵加法

    
            grads_w.insert(format!("W{}", i), dW);  // 存储 dW
            grads_b.insert(format!("b{}", i), db.clone());  // 存储 db
    
            dh = dout;
        }
    
        (grads_w, grads_b)
    }
    
    

    pub fn loss(&self, X: &Array2<f64>, y: &Array1<usize>) -> (f64, HashMap<String, Array2<f64>>, HashMap<String, Array1<f64>>) {
        let (scores, caches) = self.forward(X);
        let mut loss = 0.0;
        let mut grads_w = HashMap::new(); // 存储 dW
        let mut grads_b = HashMap::new(); // 存储 db

        match self.loss_fn {
            LossFunction::SVM => {
                let (data_loss, dout) = svm_loss(&scores, y);  // 调用 svm_loss
                loss += data_loss;
                (grads_w, grads_b) = self.backward(&dout, caches);
            }
            LossFunction::Softmax => {
                let (data_loss, dout) = softmax_loss(&scores, y);  // 调用 softmax_loss
                loss += data_loss;
                (grads_w, grads_b) = self.backward(&dout, caches);
            }
        }

        // 加入 L2 正则化的损失
        let mut reg_loss = 0.0;
        for i in 1..self.num_layers {
            reg_loss += self.reg * self.weight_params[&format!("W{}", i)].sum();
        }

        loss += reg_loss;

        (loss, grads_w, grads_b)
    }
}