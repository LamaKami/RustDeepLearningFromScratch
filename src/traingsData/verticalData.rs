use ndarray::{Array, Axis, arr0, arr1, Array2};
use ndarray_rand::rand::{self, Rng};


pub fn generate_vertical_data(classes: i32, points_per_class: i32, x_start: Option<i32>, 
    x_end: Option<i32>, y_start: Option<i32>, y_end: Option<i32>) -> (Array2<f64>, Vec<usize>){
    let x_range = x_end.unwrap_or(1) - x_start.unwrap_or(0);
    let range_per_class = x_range as f64 /classes as f64;
    let mut labels = Vec::<usize>::new();
    let mut rng = rand::thread_rng();
    let mut data_points = Array::from_elem((0, 2), 0.);

    for i in 0..classes{
        for _ in 1..points_per_class+1{
            // adding label
            labels.push(i as usize);

            // generate and adding point
            let x = rng.gen_range((range_per_class*i as f64)..(range_per_class*(i+1) as f64));
            let y = rng.gen_range(y_start.unwrap_or(0) as f64..y_end.unwrap_or(1) as f64);
            let arr= arr1(&[x,y]);
            let r = data_points.push(Axis(0), arr.view());
            let _r = if let Err(error) = r {
                panic!("Problem while pushing: {:?}", error)
            };
        }
    }

    return (data_points, labels);
}