// tests/unit/temporal-encoder.test.ts
import { describe, it, expect } from 'vitest';
import { TemporalSpikeEncoder, SpikeTrain } from '../../src/lib/memory-engine/temporal-encoder';

describe('TemporalSpikeEncoder', () => {
  it('should encode text as temporal spike trains', () => {
    const encoder = new TemporalSpikeEncoder({
      dimensions: 100,
      timeBins: 50,
      maxFiringRate: 100, // Hz
    });

    const spikeTrain = encoder.encode('hello world');

    expect(spikeTrain.spikes).toBeDefined();
    expect(spikeTrain.spikes.length).toBe(100); // dimensions
    expect(spikeTrain.spikes[0].length).toBe(50); // time bins
    expect(spikeTrain.duration).toBe(0.05); // windowDuration from default
  });

  it('should preserve temporal order in encoding', () => {
    const encoder = new TemporalSpikeEncoder();

    const train1 = encoder.encode('first');
    const train2 = encoder.encode('second');

    // Different content should produce different patterns
    expect(train1.fingerprint).not.toBe(train2.fingerprint);
  });

  it('should decode spike trains back to similarity scores', () => {
    const encoder = new TemporalSpikeEncoder();

    const original = encoder.encode('test content');
    const similar = encoder.encode('test content');
    const different = encoder.encode('completely different');

    const simScore = encoder.compare(original, similar);
    const diffScore = encoder.compare(original, different);

    expect(simScore).toBeGreaterThan(0.8);
    expect(diffScore).toBeLessThan(0.5);
  });

  it('should support batch encoding for sequences', () => {
    const encoder = new TemporalSpikeEncoder();

    const sequence = encoder.encodeSequence(['mem1', 'mem2', 'mem3']);

    expect(sequence).toHaveLength(3);
    expect(sequence[0].fingerprint).toBeDefined();
  });

  it('should extract temporal features between sequences', () => {
    const encoder = new TemporalSpikeEncoder();

    const seq1 = encoder.encodeSequence(['A', 'B']);
    const seq2 = encoder.encodeSequence(['A', 'C']);

    const temporalFeatures = encoder.extractTemporalFeatures(seq1, seq2);

    expect(temporalFeatures.commonPrefix).toBe(1); // 'A' is common
    expect(temporalFeatures.divergencePoint).toBe(1); // Diverges at index 1
  });
});
