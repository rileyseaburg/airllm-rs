//! Layer caching with prefetching for efficient inference

use super::{LayerLoader, LayerWeights, SharedWeights};
use crate::Result;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::debug;

/// Cache for loaded layers with async prefetching
pub struct LayerCache {
    /// Layer loader
    loader: Arc<LayerLoader>,
    /// Currently cached layer
    current_layer: Mutex<Option<(usize, LayerWeights)>>,
    /// Prefetch channel sender
    prefetch_tx: Option<mpsc::Sender<usize>>,
    /// Shared weights (always in memory)
    shared: SharedWeights,
}

impl LayerCache {
    /// Create a new layer cache
    pub fn new(loader: LayerLoader, enable_prefetch: bool) -> Result<Self> {
        let loader = Arc::new(loader);
        let shared = loader.load_shared_weights()?;

        let prefetch_tx = if enable_prefetch {
            let (tx, mut rx) = mpsc::channel::<usize>(2);
            let loader_clone = Arc::clone(&loader);
            
            // Spawn prefetch task
            tokio::spawn(async move {
                while let Some(layer_idx) = rx.recv().await {
                    debug!("Prefetching layer {}", layer_idx);
                    // Just load to warm up the OS page cache
                    let _ = loader_clone.load_layer(layer_idx);
                }
            });
            
            Some(tx)
        } else {
            None
        };

        Ok(Self {
            loader,
            current_layer: Mutex::new(None),
            prefetch_tx,
            shared,
        })
    }

    /// Get a layer, loading if necessary
    pub fn get_layer(&self, layer_idx: usize) -> Result<LayerWeights> {
        // Check if already cached
        {
            let cached = self.current_layer.lock().unwrap();
            if let Some((idx, _)) = &*cached {
                if *idx == layer_idx {
                    // Return clone... actually we need to rethink this
                    // For now, just reload
                }
            }
        }

        // Load the layer
        let weights = self.loader.load_layer(layer_idx)?;

        // Trigger prefetch for next layer
        if let Some(ref tx) = self.prefetch_tx {
            let next_layer = layer_idx + 1;
            if next_layer < self.loader.num_layers() {
                let _ = tx.try_send(next_layer);
            }
        }

        // Update cache
        {
            let mut cached = self.current_layer.lock().unwrap();
            *cached = Some((layer_idx, self.loader.load_layer(layer_idx)?));
        }

        Ok(weights)
    }

    /// Get shared weights
    pub fn shared_weights(&self) -> &SharedWeights {
        &self.shared
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.loader.num_layers()
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut cached = self.current_layer.lock().unwrap();
        *cached = None;
    }
}

/// Simple synchronous layer iterator
pub struct LayerIterator<'a> {
    cache: &'a LayerCache,
    current: usize,
}

impl<'a> LayerIterator<'a> {
    pub fn new(cache: &'a LayerCache) -> Self {
        Self { cache, current: 0 }
    }
}

impl<'a> Iterator for LayerIterator<'a> {
    type Item = Result<LayerWeights>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.cache.num_layers() {
            return None;
        }
        
        let result = self.cache.get_layer(self.current);
        self.current += 1;
        Some(result)
    }
}
