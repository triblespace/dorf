use std::fmt::Debug;
use std::iter::zip;
use std::marker::PhantomData;
use std::convert::AsRef;

use anndists::dist::{DistL2, Distance};
use anybytes::ByteOwner;
use digest::consts::U32;
use digest::Digest;

use tribles::{ BlobSet, Bloblike, Handle, Value};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use rand::thread_rng;
use rand::seq::SliceRandom;

#[cfg(not(target_endian = "little"))]
compile_error!(
    "This crate does not compile on BE architectures.
The reason being that most libraries just assume that they run on LE platforms,
e.g. when performing zero copy reads from transmuted arrays.
So long as Rust does not refine its handling of endianess, e.g. by introducing explicit endian
number types in core, we have no other choice than to pave the cow paths and assume
that all native numbers are little endian."
);

#[derive(AsBytes, FromZeroes, FromBytes, Debug)]
#[repr(transparent)]
pub struct Embedding<const LEN: usize, T>([T; LEN]);

impl<const LEN: usize, T> std::ops::Deref for Embedding<LEN, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const LEN: usize, T: AsBytes + Send + Sync + 'static> ByteOwner for Embedding<LEN, T> {
    fn as_bytes(&self) -> &[u8] {
        AsBytes::as_bytes(self)
    }
}

impl<const LEN: usize, T: AsBytes + Send + Sync + 'static> ByteOwner for Box<Embedding<LEN, T>> {
    fn as_bytes(&self) -> &[u8] {
        AsBytes::as_bytes(self.as_ref())
    }
}

#[derive(Debug)]
pub enum EmbeddingError {
    BadLength,
}

impl<const LEN: usize, T> TryFrom<Vec<T>> for Embedding<LEN, T> {
    type Error = Vec<T>;
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        let v = vec.try_into()?;
        Ok(Embedding(v))
    }
}

impl<'a, const LEN: usize, T> TryFrom<&'a [T]> for Embedding<LEN, T>
where
    [T; LEN]: TryFrom<&'a [T]>,
{
    type Error = EmbeddingError;

    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
        if value.len() != LEN {
            return Err(EmbeddingError::BadLength);
        }
        let Ok(arr) = value.try_into() else {
            panic!("failed conversion despite correct length")
        };
        Ok(Embedding(arr))
    }
}

pub trait Metric<T> {
    fn distance(a: &T, b: &T) -> f32;
}

pub struct MetricL2();

impl<T> Metric<T> for MetricL2
where T: AsRef<[u8]> {
    fn distance(a: &T, b: &T) -> f32 {
        DistL2::eval(&DistL2 {}, a.as_ref(), b.as_ref())
    }
}

pub struct SW<H, T, M> {
    pub blobs: BlobSet<H>,
    pub nodes: Vec<Value<Handle<H, T>>>,
    pub steps: Vec<Vec<usize>>,
    pub metric: PhantomData<M>,
}

impl<H, T, M> SW<H, T, M>
where
    H: Digest<OutputSize = U32>,
    T: Bloblike + ?Sized,
    M: Metric<T>,
{
    pub fn new(blobs: BlobSet<H>) -> Self {
        return Self {
            blobs,
            nodes: vec![],
            steps: vec![],
            metric: PhantomData,
        };
    }

    pub fn insert(&mut self, node: T) -> Value<Handle<H, T>> {
        let handle = self.blobs.insert(node);
        self.nodes.push(handle);
        handle
    }

    pub fn prepare(&mut self, random_layers: usize) {
        self.nodes.sort();
        self.nodes.dedup();

        let base: Vec<usize> = (0..self.nodes.len()).collect();
        for _ in 0..random_layers {
            let mut step = base.clone();
            step.shuffle(&mut thread_rng());
            self.steps.push(step);
        }
    }

    #[allow(unused)]
    pub fn step(&mut self) {
        if let Some(step) = self.steps.last() {
            let mut next = vec![];

            for (node_i, &target_i) in step.into_iter().enumerate() {
                let hop_i = step[target_i];
                let node = self.blobs.get(self.nodes[node_i]).unwrap().unwrap();
                let target = self.blobs.get(self.nodes[target_i]).unwrap().unwrap();
                let hop = self.blobs.get(self.nodes[hop_i]).unwrap().unwrap();

                let target_distance = M::distance(&node, &target);
                let hop_distance = M::distance(&node, &hop);

                let next_i = if hop_distance < target_distance {
                    hop_i
                } else {
                    target_i
                };

                next.push(next_i);
            }

            self.steps.push(next);
        }
    }

    pub fn multilayer_step(&mut self) -> (usize, usize) {
        if let Some(step) = self.steps.last() {
            for (i, old_step) in self.steps.iter().enumerate().rev() {
                let mut changes = 0;
                let mut next = vec![];

                for (node_i, &target_i) in step.into_iter().enumerate() {
                    let hop_i = old_step[target_i];
                    let node = self.blobs.get(self.nodes[node_i]).unwrap().unwrap();
                    let target = self.blobs.get(self.nodes[target_i]).unwrap().unwrap();
                    let hop = self.blobs.get(self.nodes[hop_i]).unwrap().unwrap();

                    let target_distance = M::distance(&node, &target);
                    let hop_distance = M::distance(&node, &hop);

                    let next_i = if hop_distance < target_distance {
                        changes += 1;
                        hop_i
                    } else {
                        target_i
                    };

                    next.push(next_i);
                }

                if changes > 0 {
                    self.steps.push(next);
                    return (i, changes);
                }
            }
        }
        (0, 0)
    }

    pub fn search(&self, query: &T, span: usize) -> Value<Handle<H, T>> {
        let mut nodes_i: Vec<(usize, f32)> = (0..span).map(|node_i| {
            let node = self.blobs.get(self.nodes[node_i]).unwrap().unwrap();
            let node_distance = M::distance(query, &node);
            (node_i, node_distance)
        }).collect();
        nodes_i.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        let mut max_distance: f32 = nodes_i.iter().map(|(_, d)| *d)
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        
        for step in self.steps.iter() {
            let mut next_nodes_i = nodes_i.clone();
            for (node_i, _) in nodes_i {
                let target_i = step[node_i];
                let target = self.blobs.get(self.nodes[target_i]).unwrap().unwrap();
    
                let target_distance = M::distance(query, &target);
    
                if target_distance < max_distance &&
                   !next_nodes_i.iter().any(|(n, _)| *n == target_i) {
                    next_nodes_i.push((target_i, target_distance));
                };
            }
            next_nodes_i.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
            next_nodes_i.truncate(span);
            max_distance = next_nodes_i.last().unwrap().1;
            nodes_i = next_nodes_i;
        }

        let node_i = nodes_i.first().unwrap().0;
        self.nodes[node_i]
    }

    #[allow(unused)]
    pub fn count_change(&mut self) -> usize {
        let l = self.steps.len();
        if l <= 1 {
            self.nodes.len()
        } else {
            let last = &self.steps[l - 1];
            let prev = &self.steps[l - 2];

            zip(last, prev).filter(|(&l, &p)| l != p).count()
        }
    }
}

/*
#[cfg(test)]
mod tests {
    //use fake::faker::name::raw::*;
    //use fake::locales::*;
    //use fake::{Dummy, Fake, Faker};
    use std::{collections::HashSet, convert::TryInto};

    //use crate::tribleset::patchtribleset::PATCHTribleSet;
    use crate::{and, find, query::ContainsConstraint, types::ShortString, ufoid, BlobSet, Id, TribleSet, NS};

    use super::*;

    NS! {
        pub namespace library {
            "47390346743AC0879BA0E77B95B9683F" as title: ShortString;
            "7B7D8B046B3FCC7CD7888C5FF03D34E8" as embedded_title: Handle<Blake3, ZC<Embedding>>;
        }
    }

    #[test]
    fn and_set() {
        let blobs = BlobSet::<Blake3>::new();
        let mut books = TribleSet::new();
        let mut book_embeddings = HNSW::new(blobs);
        let embedder = Embedder::new();

        books.union(library::entity!({
            title: ShortString::new("LOTR").unwrap(),
            embedded_title: book_embeddings.insert(embedder.embed("LOTR"))
        }));

        books.union(library::entity!({
            title: ShortString::new("Dragonrider").unwrap(),
            embedded_title: book_embeddings.insert(embedder.embed("Dragonrider"))
        }));

        books.union(library::entity!({
            title: ShortString::new("Highlander").unwrap(),
            embedded_title: book_embeddings.insert(embedder.embed("Highlander"))
        }));

        let similar: Vec<_> = find!(
            ctx,
            (title, embedding, similar_title, similar_embedding),
            and!(
                library::pattern!(ctx, books, [{
                    title: title,
                    embedding: embedding
                },
                {
                    title: similar_title,
                    embedding: similar_embedding
                }]),
                book_embeddings.k_nearest(3, embedding, similar_embedding)))
        .collect();
    }
}
*/
