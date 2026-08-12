package registry

import (
	"sync"
	"time"
)

type Worker struct {
	ID          string
	Backend     string
	Device      string
	MemoryBytes uint64
	LastSeen    time.Time
}

type Registry struct {
	mu      sync.RWMutex
	workers map[string]Worker
}

func New() *Registry {
	return &Registry{workers: make(map[string]Worker)}
}

func (r *Registry) Upsert(worker Worker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workers[worker.ID] = worker
}

func (r *Registry) Snapshot() []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workers := make([]Worker, 0, len(r.workers))
	for _, worker := range r.workers {
		workers = append(workers, worker)
	}
	return workers
}
