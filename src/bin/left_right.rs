use ndarray::{arr1, Array1};
use rand::prelude::*;
use utils::math::sigmoid;

/// The amount of nodes in a layer
const NODES_AMOUNT: usize = 10;
/// Severity of mutations
const MUTATION_RANGE: f32 = 1.0;
/// How much generations simulation lasts
const GENERATIONS: usize = 100;

#[derive(Clone)]
struct Node(Array1<f32>);

impl Node {
    /// Randomly modify a neuron
    fn mutate(&mut self) {
        let amount: f32 = random();
        let index: usize = (random::<f32>() * self.0.len() as f32).floor() as usize;
        self.0[index] = sigmoid(self.0[index] + (amount - 0.5) * MUTATION_RANGE);
    }

    fn new() -> Self {
        Self(arr1(&[0.0, 0.0]))
    }
}

struct Network(Vec<Node>);

impl Network {
    fn new() -> Self {
        let mut initial_layer: Vec<Node> = Vec::new();
        for _ in 0..NODES_AMOUNT {
            initial_layer.push(Node::new());
        }

        Self(initial_layer)
    }

    // reset a layer with updated nodes
    fn update(&mut self, set_to: Node) {
        let mut new_layer: Vec<Node> = Vec::new();
        for _ in 0..NODES_AMOUNT {
            new_layer.push(set_to.clone());
        }
        self.0 = new_layer;
    }

    /// go to the next generation
    fn gen(&mut self) {
        self.mutate_all();
        let winner = self.get_winner();
        self.update(winner);
    }

    /// Returns the node that has the most compare() precision in the generation
    fn get_winner(&self) -> Node {
        let mut winner_node = &Node::new();
        for i in &self.0 {
            if compare(&i.0) > compare(&winner_node.0) {
                winner_node = i;
            }
        }
        winner_node.clone()
    }

    /// Randomly mutates each neuron
    fn mutate_all(&mut self) {
        for i in &mut self.0 {
            i.mutate();
        }
    }
}

/// This function sets the training rules. By returning values from 0 to 1, reward is defined.
fn compare(arr: &Array1<f32>) -> f32 {
    sigmoid(arr[1] - arr[0]) 
    /* 
    * In this case, nodes with the lowest arr[0] neuron activation, 
    * and the highest arr[1] neuron activation get more reward.
    * It may look like nothing, but the network is actually dynamically
    * Adjusting itself by reverse engineering the "black box" function.
    * This simple algorithm has a ton of useful applications.
    */
}

fn main() {
    let mut net = Network::new();
    for i in 1..=GENERATIONS {
        println!("{}: {} | Arr: {:#}", i, compare(&net.0[0].0), &net.0[0].0);
        net.gen()
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn compare_test() {
//         assert_eq!(compare(&arr1(&[0., 1.])), 1.);
//         assert_eq!(compare(&arr1(&[0., 0.5])), 1., "Expected: 0.5, Got: {}", compare(&arr1(&[0.0,0.5])));
//     }
// }