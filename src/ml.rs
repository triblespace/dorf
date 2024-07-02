use std::marker::PhantomData;

use anybytes::ByteOwner;
use digest::consts::U32;
use digest::Digest;

use tribles::types::Hash;
use tribles::{BlobParseError, BlobSet, Bloblike, Bytes, Handle};
use tribles::types::hash::Blake3;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

#[derive(AsBytes, FromZeroes, FromBytes)]
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

pub struct ZeroCopy<T> {
    bytes: Bytes,
    _type: PhantomData<T>,
}

impl<T> ZeroCopy<T>
where T: ByteOwner {
    pub fn from(owner: T) -> ZeroCopy<T> {
        ZeroCopy {
            bytes: Bytes::from_owner(owner),
            _type: PhantomData
        }
    }
}

impl<T> std::ops::Deref for ZeroCopy<T>
where T: FromBytes {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        FromBytes::ref_from(&self.bytes).expect("ZeroCopy validation should happen at creation")
    }
}

impl<T> Bloblike for ZeroCopy<T>
where T: FromBytes {
    fn into_blob(self) -> Bytes {
        self.bytes
    }

    fn from_blob(blob: Bytes) -> Result<Self, BlobParseError> {
        if <T as FromBytes>::ref_from(&blob).is_none() {
            Err(BlobParseError::new("wrong size or alignment of bytes for type"))
        } else {
            Ok(ZeroCopy {bytes: blob, _type: PhantomData})
        }
    }

    fn as_handle<H>(&self) -> Handle<H, Self>
    where
        H: Digest<OutputSize = U32>,
    {
        let digest = H::digest(&self.bytes);
        unsafe { Handle::new(Hash::new(digest.into())) }
    }
}

pub enum EmbeddingError {
    BadLength
}

impl<const LEN: usize, T> TryFrom<Vec<T>> for Embedding<LEN, T> {
    type Error = EmbeddingError;
    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        let v = vec.try_into().map_err(|_| EmbeddingError::BadLength)?;
        Ok(Embedding(v))
    }
}

/*
pub struct SW<H, E, T> {
    blobs: BlobSet<H>
}

impl<H, E, T> SW<H, E, T>
where H: Digest<OutputSize = U32>{
    fn new(blobs: BlobSet<H>) -> Self {
        return Self{ blobs };
    }

    fn embed(&mut self, data: T) -> Handle<H, E> {
        let embedding = T.into();
        let handle = self.blobs.put(embedding);
        handle
    }
}
*/
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
            "7B7D8B046B3FCC7CD7888C5FF03D34E8" as embedded_title: Handle<Blake3, Embedding>;
        }
    }

    #[test]
    fn and_set() {
        let blobs = BlobSet::<Blake3>::new();
        let mut books = TribleSet::new();
        let mut book_embeddings = HNSW::new(blobs);

        books.union(library::entity!({
            title: ShortString::new("LOTR").unwrap(),
            embedded_title: book_embeddings.embed("LOTR")
        }));

        books.union(library::entity!({
            title: ShortString::new("Dragonrider").unwrap(),
            embedded_title: book_embeddings.embed("Dragonrider")
        }));

        books.union(library::entity!({
            title: ShortString::new("Highlander").unwrap(),
            embedded_title: book_embeddings.embed("Highlander")
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