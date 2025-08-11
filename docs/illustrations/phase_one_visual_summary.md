# Documentation: Phase 1 Visual Summary

This document provides a high-level visual overview of the entire data asset generation pipeline (Phase 1). It includes detailed "explainer block" diagrams for each of the three core processing scripts.

---

## 1. End-to-End Data Generation Pipeline (Phase 1)

This diagram illustrates the complete workflow, showing how we start with raw data and end with three distinct "conditioning" assets ready for the final metadata assembly.

```mermaid
graph TD
    subgraph Raw Data
        A[<img src="https://i.imgur.com/2z0e1hC.png" width="70" /><br/><b>Raw Logo Dataset</b><br/><i>/data/logos/datasetcopy</i>]
    end

    subgraph "Step 1: Cleaning"
        B(<b>data_cleaning.py</b><br/><i>Verifies image integrity</i>)
    end

    subgraph "Verified Asset"
        C[<img src="https://i.imgur.com/2z0e1hC.png" width="70" /><br/><b>Cleaned Logo Dataset</b><br/><i>/data/logos/cleaned</i>]
    end

    subgraph "Step 2: Conditioning Data Generation (Parallel)"
        D(<b>1_create_sketches.py</b><br/><i>Generates line art</i>)
        E(<b>2_extract_colors.py</b><br/><i>Finds dominant colors</i>)
        F(<b>3_generate_captions.py</b><br/><i>Creates text descriptions</i>)
    end

    subgraph "Final Conditioning Assets"
        G[<img src="https://i.imgur.com/sISi2Zg.png" width="70" /><br/><b>Sketch Dataset</b><br/><i>/data/logos/sketches</i>]
        H[<img src="https://i.imgur.com/O3bPLcW.png" width="70" /><br/><b>Color Palettes</b><br/><i>/data/color_palettes.json</i>]
        I[<img src="https://i.imgur.com/j8kLqYd.png" width="70" /><br/><b>Captions</b><br/><i>/data/captions.json</i>]
    end

    A --> B;
    B --> C;
    C --> D;
    C --> E;
    C --> F;
    D --> G;
    E --> H;
    F --> I;

    style A fill:#f9f9f9,stroke:#333,stroke-width:2px
    style C fill:#e6ffed,stroke:#333,stroke-width:2px
    style G fill:#e6f7ff,stroke:#333,stroke-width:2px
    style H fill:#e6f7ff,stroke:#333,stroke-width:2px
    style I fill:#e6f7ff,stroke:#333,stroke-width:2px
```

---

## 2. Explainer Block Diagrams

These diagrams provide a more detailed, "tech-company-style" look into the internal logic of each script.

### **Script 1: `1_create_sketches.py`**

<br/>

```mermaid
graph TD
    subgraph "Input: Single Cleaned Image"
        InputImg[<img src="https://i.imgur.com/2z0e1hC.png" width="100" /><br/><i>e.g., /cleaned/train/Food/Oreo/1.jpg</i>]
    end

    subgraph "Core Transformation: Sketch Algorithm"
        direction LR
        subgraph "Image Processing Pipeline"
            direction TB
            A[Convert to Grayscale] --> B[Invert Colors];
            B --> C[Apply Gaussian Blur];
            C --> D[Invert Colors Again];
        end
        subgraph "Final Blend"
            direction TB
            A_copy((Grayscale Image))
            D_copy((Blurred Inverted Image))
            A_copy --> E{Color Dodge Blend};
            D_copy --> E;
        end
    end
    
    subgraph "Output: Sketch Image"
        OutputImg[<img src="https://i.imgur.com/sISi2Zg.png" width="100" /><br/><i>e.g., /sketches/train/Food/Oreo/1.jpg</i>]
    end

    InputImg --> A;
    E --> OutputImg;

    style InputImg fill:#f9f9f9,stroke:#333,stroke-width:2px
    style OutputImg fill:#f9f9f9,stroke:#333,stroke-width:2px
```

### **Script 2: `2_extract_colors.py`**

<br/>

```mermaid
graph TD
    subgraph "Input: Single Cleaned Image"
        InputImg[<img src="https://i.imgur.com/2z0e1hC.png" width="100" /><br/><i>e.g., /cleaned/train/Food/Oreo/1.jpg</i>]
    end

    subgraph "Core Transformation: K-Means Clustering"
        A[Resize & Convert to Pixel List] --> B{Plot Pixels in 3D RGB Space};
        B --> C[Find 5 Cluster Centroids (K-Means)];
        C --> D[Extract Centroid RGB Values];
    end
    
    subgraph "Output: JSON Data Entry"
        OutputJSON[<img src="https://i.imgur.com/O3bPLcW.png" width="100" /><br/><i>"train/Food/Oreo/1.jpg":<br/>["#000000", "#ffffff", ...]</i>]
    end

    InputImg --> A;
    D --> OutputJSON;

    style InputImg fill:#f9f9f9,stroke:#333,stroke-width:2px
    style OutputJSON fill:#f9f9f9,stroke:#333,stroke-width:2px
```

### **Script 3: `3_generate_captions.py`**

<br/>

```mermaid
graph TD
    subgraph "Input: Single Cleaned Image"
        InputImg[<img src="https://i.imgur.com/2z0e1hC.png" width="100" /><br/><i>e.g., /cleaned/train/Food/Oreo/1.jpg</i>]
    end

    subgraph "Core Transformation: BLIP Model Inference"
        A[Pre-process Image to Tensor] --> B{BLIP Model};
        B -- Generates --> C[Sequence of Token IDs];
        C --> D[Decode IDs to Text];
    end
    
    subgraph "Output: JSON Data Entry"
        OutputJSON[<img src="https://i.imgur.com/j8kLqYd.png" width="100" /><br/><i>"train/Food/Oreo/1.jpg":<br/>"a black and white logo..."</i>]
    end

    InputImg --> A;
    D --> OutputJSON;

    style InputImg fill:#f9f9f9,stroke:#333,stroke-width:2px
    style OutputJSON fill:#f9f9f9,stroke:#333,stroke-width:2px
```
