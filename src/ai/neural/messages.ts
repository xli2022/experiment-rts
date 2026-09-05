/**
 * What the main thread and the inference worker say to each other.
 *
 * One `init` per worker, then one `act` per decision, each answered by id.
 * The arrays in an `act` are transferred rather than copied: an observation is
 * about 150 KB and the agent allocates a fresh one per decision precisely so
 * it can be given away.
 */

export interface InitMessage {
  readonly type: 'init';
  readonly modelUrl: string;
  readonly wasmUrl: string;
  readonly numThreads: number;
}

export interface ActMessage {
  readonly type: 'act';
  readonly id: number;
  readonly entities: Float32Array;
  readonly entityMask: Uint8Array;
  readonly grid: Float32Array;
  readonly scalars: Float32Array;
  readonly maskType: Uint8Array;
  readonly maskSelection: Uint8Array;
  readonly maskTarget: Uint8Array;
  readonly maskCell: Uint8Array;
  readonly maskBuildCell: Uint8Array;
  readonly maskRowEntityType: Uint8Array;
  readonly maskBuildType: Uint8Array;
  readonly temperature: number;
}

export type ToWorker = InitMessage | ActMessage;

export interface ReadyMessage {
  readonly type: 'ready';
  /** How long the warm-up inference took. */
  readonly warmupMs: number;
}

export interface ResultMessage {
  readonly type: 'result';
  readonly id: number;
  /** `ACTION_INTS` integers; see `actionFromInts`. */
  readonly action: Int32Array;
  readonly ms: number;
}

export interface ErrorMessage {
  readonly type: 'error';
  /** The request that failed, or absent when the worker itself did. */
  readonly id?: number;
  readonly message: string;
}

export type FromWorker = ReadyMessage | ResultMessage | ErrorMessage;
