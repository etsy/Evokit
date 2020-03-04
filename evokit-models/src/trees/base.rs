#[derive(Debug, Clone, Deserialize, Serialize)]
/// We encode trees as a flattened binary tree.  It contains 2 fields:
/// [feature_index, feature_split_value].  We store the payload values
/// in a separate field to allow specialization for later.
pub(super) struct BaseTree {
    /// Number of levels in the tree
    levels: u32,

    /// We store features separately from splits
    pub features: Vec<usize>,

    /// Value of the feature to split at
    pub splits: Vec<f32>,

    /// payloads
    pub values: Vec<f32>,
}

impl BaseTree {
    /// Initializes a tree of zeros
    pub fn zero(levels: u32) -> Self {
        BaseTree {
            levels: levels as u32,
            features: vec![0; 2usize.pow(levels as u32) - 1],
            splits: vec![0.; 2usize.pow(levels as u32) - 1],
            values: vec![0.; 2usize.pow(levels as u32)],
        }
    }

    /// Given a dense vector, outputs the final prediction from this tree
    pub fn predict(&self, data: &[f32]) -> f32 {
        // Get the levels
        let mut index = 0usize;
        for _ in 0..self.levels {
            index = 2 * index
                + if data[self.features[index]] <= self.splits[index] {
                    1
                } else {
                    2
                };
        }

        // Get the offset into values
        let val_offset = index - (2usize.pow(self.levels) - 1);
        self.values[val_offset as usize]
    }
}

#[cfg(test)]
mod tree_base_tree {
    use super::*;

    fn build_tree() -> BaseTree {
        // Two levels
        let mut tree = BaseTree::zero(2);
        assert_eq!(tree.splits.len(), 3);
        assert_eq!(tree.values.len(), 4);

        tree.features = vec![0, 1, 1];
        tree.splits = vec![0.5, 0.7, 0.9];
        tree.values = vec![0.1, 0.2, 0.3, 0.4];

        tree
    }

    #[test]
    fn test_trees() {
        let tree = build_tree();

        // Should go left then right
        let data = vec![0.4, 0.8];
        let result = tree.predict(&data);
        assert_eq!(0.2, result);

        // Should go right then left
        let data = vec![6., 0.8];
        let result = tree.predict(&data);
        assert_eq!(0.3, result);

        // Should go right then right
        let data = vec![5.5, 0.91];
        let result = tree.predict(&data);
        assert_eq!(0.4, result);

        // Should go left then left, testing lte
        let data = vec![0.5, 0.7];
        let result = tree.predict(&data);
        assert_eq!(0.1, result);
    }
}
