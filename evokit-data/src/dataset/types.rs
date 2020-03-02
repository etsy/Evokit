use hashbrown;

use std::convert::TryInto;
use std::fmt::Debug;

use self::hashbrown::HashMap;
use crate::datatypes::Sparse;

/// Defines datastypes

#[derive(Deserialize, Clone, Debug, PartialEq)]
/// Types of metadata
pub enum MetaType {
    /// Numeric
    Num(f32),
    /// String
    Str(String),
}

impl TryInto<f32> for &MetaType {
    type Error = &'static str;
    fn try_into(self) -> Result<f32, Self::Error> {
        match self {
            MetaType::Num(val) => Ok(*val),
            _ => Err("Expected type to be Num"),
        }
    }
}

/// Metadata
pub type Metadata = HashMap<String, MetaType>;
/// Sparse datatype

/// The trait for parsing different data types
pub trait DataParse: Sync {
    /// Output type
    type Out: Debug;

    /// Parses one line into Metadata and the output type
    fn parse<'a, I: Iterator<Item = &'a str>>(&self, xs: I) -> Option<(Metadata, Self::Out)>;
}

#[derive(Debug, Clone)]
/// Defines the parser for dense data
pub struct DenseData;

impl DataParse for DenseData {
    type Out = Vec<f32>;

    fn parse<'a, I: Iterator<Item = &'a str>>(&self, xs: I) -> Option<(Metadata, Self::Out)> {
        let v: Option<Vec<_>> = xs
            .map(|x| x.split(':').last().and_then(|x| x.parse().ok()))
            .collect();

        v.map(|feats| (HashMap::new(), feats))
    }
}

#[derive(Debug, Clone)]
/// Defines the parser for sparse data
pub struct SparseData(pub usize);

impl DataParse for SparseData {
    type Out = Sparse;

    fn parse<'a, I: Iterator<Item = &'a str>>(&self, xs: I) -> Option<(Metadata, Self::Out)> {
        let mut hm = HashMap::new();
        let mut iv: Vec<(usize, f32)> = Vec::new();
        for xi in xs {
            let mut p = xi.split(':');
            if let (Some(name), Some(val)) = (p.next(), p.next()) {
                match (name.parse(), val.parse()) {
                    (Ok(idx), Ok(v)) => iv.push((idx, v)),

                    (Ok(_), _) => return None,

                    (_, Ok(v)) => {
                        hm.insert(name.to_string(), MetaType::Num(v));
                    }

                    _ => {
                        hm.insert(name.to_string(), MetaType::Str(val.to_string()));
                    }
                };
            }
        }
        // Sort then dedup by key
        iv.sort_by_key(|x| x.0);
        iv.dedup_by_key(|x| x.0);
        let (mut is, mut vs): (Vec<_>, Vec<_>) = iv
            .into_iter()
            .filter(|x| x.0 < self.0 && x.1 != 0.0)
            .unzip();

        hm.shrink_to_fit();
        is.shrink_to_fit();
        vs.shrink_to_fit();
        Some((hm, Sparse(self.0, is, vs)))
    }
}

#[derive(Debug, Clone)]
/// Converts dense to sparse
pub struct DenseFromSparseData(pub usize);

impl DataParse for DenseFromSparseData {
    type Out = Vec<f32>;
    fn parse<'a, I: Iterator<Item = &'a str>>(&self, xs: I) -> Option<(Metadata, Self::Out)> {
        SparseData(self.0)
            .parse(xs)
            .map(|(ns, sd)| (ns, sd.to_dense().0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let s = "name:hello world:1.23 1:23 3:21 2:13";
        let sd = SparseData(10);
        let (hm, sparse) = sd.parse(s.split_whitespace()).expect("Should have worked!");
        let mut expected_hm = HashMap::new();
        expected_hm.insert("name".into(), MetaType::Str("hello".into()));
        expected_hm.insert("world".into(), MetaType::Num(1.23));

        assert_eq!(expected_hm, hm);
        assert_eq!(sparse.0, 10);
        assert_eq!(sparse.1, vec![1, 2, 3]);
        assert_eq!(sparse.2, vec![23., 13., 21.]);
    }
}
