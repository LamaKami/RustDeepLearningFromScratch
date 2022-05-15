use ndarray::{arr2, Array, Array2, Axis, ArrayBase, ViewRepr, Dim};
use itertools::izip;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
mod traingsData;
use traingsData::verticalData::generate_vertical_data;

fn main() {
    let (inputs, labels) = generate_vertical_data(3, 50, None, None, None, None);
    //let inputs = arr2(&[[0.0, 0.0],[0.10738789, 0.02852226],[ 0.09263825, -0.20199226],[-0.32224888, -0.08524539],[-0.3495118, 0.27454028],[-0.52100587, 0.19285966],[0.5045865,  0.43570277],[0.76882404, 0.11767714],[ 0.49269393, -0.73984873],[-0.70364994, -0.71054685],[-0., -0.],[-0.07394107,  0.08293611],[0.00808054, 0.22207525],[0.24548167, 0.22549914],[ 0.38364738, -0.22437814],[-0.00801609, -0.5554977 ],[-0.66060567, 0.08969161],[-0.7174548, 0.30032802],[0.17299275, 0.87189275],[0.66193414, 0.74956197],[-0.0, 0.0],[ 0.05838184, -0.09453698],[-0.13682534, -0.17510438],[-0.27516943, -0.18812999],[0.19194843, 0.40085742],[-0.16649488, 0.53002024],[0.6666014,  0.00932745],[ 0.43282092, -0.6462231 ],[-0.87291753, -0.16774514],[-0.6297623,0.77678794]]);
    //let labels: [usize; 30] = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2];
    let mut dense1 = LayerDense::new(2,3);
    let mut relu = ActivationReLu::new();
    let mut dense2 = LayerDense::new(3,3);
    let mut softmax = ActivationSoftmax::new();
    // i need a init function which checks if the last n_neurons are the same as the unique numbers in labels, because if not so we get an index error when we calculating the loss

    // Helper variables
    let mut lowest_loss = 99999999.;
    let mut best_dense1_weights = dense1.weights.clone();
    let mut best_dense1_biases = dense1.biases.clone();
    let mut best_dense2_weights = dense2.weights.clone();
    let mut best_dense2_biases = dense2.biases.clone();


    for it in 1..4000{

        // Update weights and biases for iteration with small values
        dense1.weights = dense1.weights + (0.05 * Array::random((2, 3), StandardNormal));
        dense1.biases = dense1.biases + (0.05 * Array::random((1, 3), StandardNormal));
        dense2.weights = dense2.weights + (0.05 * Array::random((3, 3), StandardNormal));
        dense2.biases = dense2.biases + (0.05 * Array::random((1, 3), StandardNormal));


        dense1.forward(&inputs);
        relu.forward(&dense1.output.as_ref().unwrap());
        dense2.forward(relu.output.as_ref().unwrap());
        softmax.forward(&dense2.output.as_ref().unwrap());
    
        let loss = LossCategoricalCrossentropy::new(softmax.output.as_ref().unwrap().to_owned(), labels.to_vec());
        let calc_loss = loss.calculate();
        println!("iteration: {}, loss: {}, acc {}",it, calc_loss, accuracy(softmax.output.as_ref().unwrap(),&labels));
        //println!("{:?}",softmax.output.as_ref().unwrap());
        if calc_loss < lowest_loss{
            lowest_loss = calc_loss;
            best_dense1_weights = dense1.weights.clone();
            best_dense1_biases = dense1.biases.clone();
            best_dense2_weights = dense2.weights.clone();
            best_dense2_biases = dense2.biases.clone();
        } else {
            //reset
            dense1.weights = best_dense1_weights.clone();
            dense1.biases = best_dense1_biases.clone();
            dense2.weights = best_dense2_weights.clone();
            dense2.biases = best_dense2_biases.clone();
        }
    }
    
}

// layer Part
#[derive(Debug)]
pub struct LayerDense{
    pub biases: Array2<f64>,
    pub weights: Array2<f64>,
    pub output: Option<Array2<f64>>
}

impl LayerDense{
    fn new(n_inputs: usize, n_neurons: usize) -> LayerDense{
        LayerDense{
            biases: Array::zeros((1, n_neurons)),
            // We are mult with 0.01 that we have small values but not 0, if they are to large the model takes longer to fit (S. 67-68)
            weights: 0.01 * Array::random((n_inputs, n_neurons), StandardNormal),
            output: None
        }
    }

    fn forward(&mut self, inputs: &Array2<f64>){
        self.output = Some(inputs.dot(&self.weights) + &self.biases);
    }
}

// Activation Part

pub struct ActivationReLu{
    pub output: Option<Array2<f64>>
}

impl ActivationReLu{
    fn forward(&mut self, inputs: &Array2<f64>){
        self.output = Some(inputs.map(|x| max(x.to_owned(), 0.0)));
    }

    fn new() -> ActivationReLu{
        ActivationReLu{
            output: None
        }
    }
}

pub struct ActivationSoftmax{
    pub output: Option<Array2<f64>>
}

impl ActivationSoftmax{
    fn forward(&mut self, inputs: &Array2<f64>){
        let mut output = Array::from_elem((0, inputs.shape()[1]), 0.);
        for row in inputs.axis_iter(Axis(0)){
            let max_row_element = max_val(row);
            let exp_values = row.map(|x| (x - max_row_element).exp());
            let normalizer = exp_values.sum();
            let result = exp_values.map(|x| x/normalizer);
            let r = output.push(Axis(0), result.view());

            let _r = match r {
                Ok(_r) => (),
                Err(error) => panic!("Problem while pushing: {:?}", error),
            };
            
        }
        self.output = Some(output);
    }

    fn new() -> ActivationSoftmax{
        ActivationSoftmax{
            output: None
        }
    }
}


fn max(v1: f64, v2: f64) -> f64{
    if v1 > v2{
        return v1;
    }
    v2
}

fn max_val(arr: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>) -> f64{
    if arr.len() == 0{
        panic!("Output values need at least one prediction")
    }
    let mut largest = arr[0];
    for val in arr{
        if val > &largest{
            largest = *val;
        }
    }
    return largest;
}


// Loss Part
pub struct LossCategoricalCrossentropy{
    output: Array2<f64>,
    y: Vec<usize>
}

trait Loss{
    fn new(output: Array2<f64>, y: Vec<usize>) -> Self;

    fn calculate(&self) -> f64;

    fn forward(y_pred: &Array2<f64>, y_true: &Vec<usize>) ->  Vec<f64>;
}

impl Loss for LossCategoricalCrossentropy{
    fn new(output: Array2<f64>, y: Vec<usize>) -> Self {
        LossCategoricalCrossentropy { output, y }
    }

    fn calculate(&self) -> f64{
        let sample_loss = Self::forward(&self.output, &self.y);
        mean(&sample_loss)
    }

    fn forward(y_pred: &Array2<f64>, y_true: &Vec<usize>) -> Vec<f64> {
        let y_pred_clipped = clip(y_pred, 1.0e-7, 1.0 - 1.0e-7);
        
        let mut correct_confidence = Vec::new();

        for (predictions, correct_label) in izip!(y_pred_clipped.rows(), y_true){
            correct_confidence.push(-predictions[correct_label.to_owned()].ln());
        }

        //print!("{:?}", correct_confidence);
        return correct_confidence;
    }
}

fn clip(values: &Array2<f64>, lower: f64, upper: f64) -> Array2<f64>{
    return values.map(|x| -> f64 {
        if x>&upper{
            return upper
        } else if x< &lower {
            return lower
        }
        *x
    });
}

fn mean(list: &Vec<f64>) -> f64 {
    let sum: f64 = Iterator::sum(list.iter());
    f64::from(sum) / (list.len() as f64)
}

fn accuracy(predictions: &Array2<f64>, labels: &Vec<usize>) -> f64{
    let mut hits = 0.;
    for (prediction, label) in izip!(predictions.rows(), labels){
        //println!("predictions: {:?}   labels {:?}",prediction, label);

        if max_val(prediction) == prediction[label.to_owned()]{
            hits += 1.;
        }
    }
    hits/labels.len() as f64
}   