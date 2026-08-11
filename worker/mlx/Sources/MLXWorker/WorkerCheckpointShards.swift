enum WorkerCheckpointShards {
    static let defaultModelID = "mlx-community/gemma-3-270m-it-4bit"

    static let configuration = CheckpointShardRuntimeConfiguration(
        adapterRegistry: CheckpointShardAdapterRegistry([
            Gemma3CheckpointShardAdapter()
        ]),
        workerBudgetBytes: 128 * 1024 * 1024,
        maxOpenSequencesPerShard: 16,
        tokens: [1, 2, 3, 4, 5, 6],
        rtol: 1e-4,
        atol: 1e-4
    )
}
