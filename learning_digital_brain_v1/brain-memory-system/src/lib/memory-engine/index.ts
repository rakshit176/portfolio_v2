// =============================================================================
// Brain-Inspired Memory Engine — Unified Entry Point
//
// Hippocampal Page Index Architecture:
//
//   Input → [Working Memory] → [Pattern Separator (DG)] → [Page Index Creator]
//                                                                  ↓
//                                                       [CA3 Hopfield Network]
//                                                                  ↓
//                                                       Index → Neocortex Map
//                                                                  ↓
//                                                       [Neocortical Store]
//                                                                  ↓
//                                                       [Consolidation Engine]
// =============================================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type {
  MemoryItem,
  PageIndex,
  RecallResult,
  WorkingMemoryItem,
  MemorySystemStats,
  ConsolidationResult,
  MemoryGraphEdge,
  RecallOptions,
  MemorySystemOptions,
} from "./types";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export { MEMORY_CONFIG } from "./config";
export type { MemoryConfig } from "./config";

// ---------------------------------------------------------------------------
// Subsystems
// ---------------------------------------------------------------------------

export { PatternSeparator, createPRNG, hashToVector } from "./dentate-gyrus";

export { HopfieldNetwork } from "./hopfield";
export type { HopfieldRecallResult } from "./hopfield";

export {
  HippocampalPageIndex,
} from "./page-index";
export type { PageIndexLookupResult, PageIndexStats } from "./page-index";

export { NeocortexStore } from "./neocortex-store";
export type { SimilaritySearchResult } from "./neocortex-store";

export { ConsolidationEngine } from "./consolidation";

export { WorkingMemory } from "./working-memory";

export { STDPModule } from "./stdp";
export type { STDPAssociation } from "./stdp";

export {
  TemporalSpikeEncoder,
} from "./temporal-encoder";
export type { SpikeTrain, TemporalFeatures, EncoderConfig } from "./temporal-encoder";

// ---------------------------------------------------------------------------
// Main System
// ---------------------------------------------------------------------------

export { BrainMemorySystem } from "./memory-system";

// ---------------------------------------------------------------------------
// Demo Data Generator
// ---------------------------------------------------------------------------

import { BrainMemorySystem } from "./memory-system";
import { MemoryItem } from "./types";

/**
 * Demo data definition: content and metadata for sample memories.
 */
interface DemoMemoryDef {
  content: string;
  category: string;
  tags?: string[];
}

/**
 * Pre-defined demo memories spanning multiple knowledge domains.
 * Designed to showcase pattern separation (different categories → different codes)
 * and consolidation (repeated recall of some memories).
 */
const DEMO_MEMORIES: DemoMemoryDef[] = [
  // Neuroscience (4 memories)
  {
    content:
      "The hippocampus creates sparse index codes for efficient memory retrieval",
    category: "neuroscience",
    tags: ["hippocampus", "memory", "indexing"],
  },
  {
    content:
      "Dentate gyrus pattern separation ensures similar inputs produce orthogonal representations",
    category: "neuroscience",
    tags: ["dentate gyrus", "pattern separation", "orthogonality"],
  },
  {
    content:
      "CA3 recurrent connections enable autoassociative pattern completion in the hippocampus",
    category: "neuroscience",
    tags: ["CA3", "hopfield", "pattern completion"],
  },
  {
    content:
      "Sleep-dependent consolidation transfers memories from hippocampus to neocortex over time",
    category: "neuroscience",
    tags: ["consolidation", "sleep", "neocortex"],
  },

  // Physics (3 memories)
  {
    content:
      "Einstein's theory of relativity unified space and time into a single spacetime continuum",
    category: "physics",
    tags: ["relativity", "spacetime", "einstein"],
  },
  {
    content:
      "Quantum entanglement creates correlations between particles regardless of distance",
    category: "physics",
    tags: ["quantum", "entanglement", "particles"],
  },
  {
    content:
      "The Heisenberg uncertainty principle limits simultaneous measurement of position and momentum",
    category: "physics",
    tags: ["quantum", "uncertainty", "measurement"],
  },

  // Computer Science (3 memories)
  {
    content:
      "Hash tables provide average O(1) lookup time by mapping keys to array indices",
    category: "computer science",
    tags: ["hash tables", "algorithms", "complexity"],
  },
  {
    content:
      "Neural networks learn hierarchical representations through backpropagation",
    category: "computer science",
    tags: ["neural networks", "deep learning", "backpropagation"],
  },
  {
    content:
      "Bloom filters provide space-efficient probabilistic membership testing",
    category: "computer science",
    tags: ["bloom filters", "probability", "data structures"],
  },

  // Mathematics (3 memories)
  {
    content:
      "The Fourier transform decomposes signals into constituent frequencies",
    category: "mathematics",
    tags: ["fourier", "signals", "frequencies"],
  },
  {
    content:
      "Euler's identity connects five fundamental constants: e, i, pi, 1, and 0",
    category: "mathematics",
    tags: ["euler", "identity", "constants"],
  },
  {
    content:
      "Graph theory studies networks of nodes and edges with applications in social networks and routing",
    category: "mathematics",
    tags: ["graph theory", "networks", "topology"],
  },

  // Biology (3 memories)
  {
    content:
      "DNA double helix structure was discovered by Watson and Crick in 1953",
    category: "biology",
    tags: ["DNA", "genetics", "discovery"],
  },
  {
    content:
      "Mitochondria produce ATP through oxidative phosphorylation in cellular respiration",
    category: "biology",
    tags: ["mitochondria", "ATP", "respiration"],
  },
  {
    content:
      "Synaptic plasticity allows neurons to strengthen or weaken connections based on experience",
    category: "biology",
    tags: ["synapses", "plasticity", "learning"],
  },

  // Philosophy (2 memories)
  {
    content:
      "Descartes' cogito ergo sum established the thinking self as the foundation of knowledge",
    category: "philosophy",
    tags: ["descartes", "cogito", "epistemology"],
  },
  {
    content:
      "The Chinese room argument challenges the notion that computation alone can produce understanding",
    category: "philosophy",
    tags: ["chinese room", "consciousness", "AI"],
  },
];

/**
 * Generate demo data by populating a BrainMemorySystem with sample memories.
 *
 * Creates 18 memories across 6 categories (neuroscience, physics, computer science,
 * mathematics, biology, philosophy), then simulates some recalls to demonstrate
 * consolidation dynamics.
 *
 * @param system Optional existing system to populate. If not provided, a new one is created.
 * @returns The system populated with demo data.
 */
export function generateDemoData(
  system?: BrainMemorySystem,
): {
  system: BrainMemorySystem;
  memories: MemoryItem[];
} {
  const memory = system ?? new BrainMemorySystem({ seed: 42 });

  // Store all demo memories
  const memories: MemoryItem[] = [];
  for (const def of DEMO_MEMORIES) {
    const item = memory.store(def.content, {
      category: def.category,
      tags: def.tags,
    });
    memories.push(item);
  }

  // Simulate some recalls to kick-start consolidation
  // Recall the first few neuroscience memories multiple times
  const frequentlyRecalled = [
    "hippocampus sparse codes retrieval",
    "pattern separation orthogonal representations",
    "CA3 recurrent pattern completion hippocampus",
  ];

  for (let i = 0; i < 3; i++) {
    for (const query of frequentlyRecalled) {
      memory.recall(query);
    }
  }

  // Run a consolidation cycle
  memory.runConsolidation();

  // Add some items to working memory
  memory.addToWorkingMemory(
    "User is exploring the memory system demo",
    0.9,
  );
  memory.addToWorkingMemory(
    "Current topic: neuroscience of memory formation",
    0.7,
  );

  return { system: memory, memories };
}
