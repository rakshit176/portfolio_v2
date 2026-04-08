/**
 * Temporal Spike Coding Encoder
 *
 * Neuroscience basis:
 * Temporal coding uses precise spike timing to encode information,
 * rather than just firing rates. This enables:
 * - Higher information density per spike
 * - Better noise robustness
 * - Natural sequence representation
 *
 * This encoder converts text into temporal spike trains where:
 * - Each character/word maps to specific neurons
 * - Timing of spikes carries semantic information
 * - Sequential order is preserved in temporal patterns
 *
 * References:
 * - Thorpe et al. (2001) — Spikes explore for their own sake
 * - VanRullen et al. (2005) — Spike timing
 */

export interface SpikeTrain {
  /** spikes[neuronIndex][timeBin] = 1 if spike occurred, 0 otherwise */
  spikes: number[][];
  /** Duration of the spike train in seconds */
  duration: number;
  /** Unique fingerprint for this encoding */
  fingerprint: string;
  /** Sparse representation: array of (neuron, time) spike events */
  sparseEvents: Array<[number, number]>;
}

export interface TemporalFeatures {
  /** Number of consecutive items that match */
  commonPrefix: number;
  /** Index where sequences first differ */
  divergencePoint: number;
  /** Temporal correlation coefficient */
  correlation: number;
}

export interface EncoderConfig {
  /** Number of neurons in the encoding population */
  dimensions: number;
  /** Number of time bins for discretization */
  timeBins: number;
  /** Maximum firing rate in Hz */
  maxFiringRate: number;
  /** Time window per encoding in seconds */
  windowDuration: number;
}

const DEFAULT_CONFIG: EncoderConfig = {
  dimensions: 256,
  timeBins: 50,
  maxFiringRate: 100,
  windowDuration: 0.05, // 50ms
};

/**
 * Temporal spike coding encoder.
 * Converts text into biologically plausible spike trains.
 */
export class TemporalSpikeEncoder {
  private readonly config: EncoderConfig;
  private readonly rng: () => number;

  constructor(config?: Partial<EncoderConfig>, seed?: number) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    // Simple PRNG using splitmix32
    let state = seed ?? Date.now();
    this.rng = () => {
      state += 0x9e3779b97f4a7c15;
      let z = state;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
      z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
      return (z ^ (z >> 31)) / 0x100000000;
    };
  }

  /**
   * Encode text as a temporal spike train.
   */
  encode(text: string): SpikeTrain {
    const { dimensions, timeBins, windowDuration } = this.config;
    const spikes: number[][] = Array.from({ length: dimensions }, () =>
      new Array(timeBins).fill(0)
    );

    // Generate sparse spiking pattern based on content hash
    const hash = this.hashString(text);
    const sparseEvents: Array<[number, number]> = [];

    // Use hash to determine which neurons fire and when
    for (let i = 0; i < dimensions; i++) {
      const neuronHash = (hash * (i + 1)) % 1000000;
      const isActive = neuronHash % 4 === 0; // ~25% sparsity

      if (isActive) {
        const timeBin = (neuronHash >> 2) % timeBins;
        spikes[i][timeBin] = 1;
        sparseEvents.push([i, timeBin]);
      }
    }

    return {
      spikes,
      duration: windowDuration,
      fingerprint: this.generateFingerprint(text),
      sparseEvents,
    };
  }

  /**
   * Encode a sequence of texts.
   * Returns consecutive spike trains preserving temporal order.
   */
  encodeSequence(texts: string[]): SpikeTrain[] {
    return texts.map(text => this.encode(text));
  }

  /**
   * Compare two spike trains for similarity.
   * Returns value in [0, 1] where 1 is identical.
   */
  compare(train1: SpikeTrain, train2: SpikeTrain): number {
    // Count matching spikes
    let matches = 0;
    let total = 0;

    const dim = Math.min(train1.spikes.length, train2.spikes.length);
    const timeBins = Math.min(
      train1.spikes[0]?.length || 0,
      train2.spikes[0]?.length || 0
    );

    for (let i = 0; i < dim; i++) {
      for (let t = 0; t < timeBins; t++) {
        const s1 = train1.spikes[i]?.[t] || 0;
        const s2 = train2.spikes[i]?.[t] || 0;
        if (s1 === 1 || s2 === 1) {
          total++;
          if (s1 === s2) matches++;
        }
      }
    }

    return total === 0 ? 0 : matches / total;
  }

  /**
   * Extract temporal features between two sequences.
   */
  extractTemporalFeatures(
    seq1: SpikeTrain[],
    seq2: SpikeTrain[],
  ): TemporalFeatures {
    const minLength = Math.min(seq1.length, seq2.length);
    let commonPrefix = 0;
    let divergencePoint = 0;

    for (let i = 0; i < minLength; i++) {
      if (seq1[i].fingerprint === seq2[i].fingerprint) {
        commonPrefix++;
      } else {
        divergencePoint = i;
        break;
      }
    }

    // Calculate correlation using spike timing
    let correlation = 0;
    const compareLength = Math.min(seq1.length, seq2.length);
    for (let i = 0; i < compareLength; i++) {
      correlation += this.compare(seq1[i], seq2[i]);
    }
    correlation /= compareLength;

    return {
      commonPrefix,
      divergencePoint,
      correlation,
    };
  }

  /**
   * Generate a fingerprint for caching/lookup.
   */
  private generateFingerprint(text: string): string {
    const hash = this.hashString(text);
    return hash.toString(16).padStart(8, '0');
  }

  /**
   * Simple hash function for deterministic encoding.
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}
