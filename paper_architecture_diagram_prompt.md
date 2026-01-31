# DWTS投票系统分析论文架构图生成Prompt

## Style Reference & Execution Instructions

### 1. Art Style (Visio/Illustrator Aesthetic)

Generate a **professional academic architecture diagram** suitable for a top-tier mathematical modeling competition paper (MCM/ICM).

* **Visuals:** Flat vector graphics, distinct geometric shapes, clean thin outlines, and soft pastel fills (Azure Blue, Slate Grey, Coral Orange, Forest Green).

* **Layout:** Strictly follow the spatial arrangement defined below.

* **Vibe:** Technical, precise, clean white background. NOT hand-drawn, NOT photorealistic, NOT 3D render, NO shadows/shading.

### 2. CRITICAL TEXT CONSTRAINTS (Read Carefully)

* **DO NOT render meta-labels:** Do not write words like "ZONE 1", "LAYOUT CONFIGURATION", "Input", "Output", or "Container" inside the image. These are structural instructions for YOU, not text for the image.

* **ONLY render "Key Text Labels":** Only text inside double quotes (e.g., "[Text]") listed under "Key Text Labels" should appear in the diagram.

* **Font:** Use a clean, bold Sans-Serif font (like Roboto or Helvetica) for all labels.

### 3. Visual Schema Execution

Translate the following structural blueprint into the final image:

---

## BEGIN PROMPT

**[Style & Meta-Instructions]**

High-fidelity scientific schematic, technical vector illustration, clean white background, distinct horizontal boundaries separating layers, academic textbook style. High resolution 4K, strictly 2D flat design with subtle flow arrows connecting stages. Use professional, distinct color coding for each stage.

**[LAYOUT CONFIGURATION]**

Selected Layout: Sequential Pipeline Architecture with Data Flow.

Composition Logic: Five distinct horizontal stages arranged left-to-right in a pipeline. Each stage processes data and passes results to the next stage. Top layer shows data flow, middle layer shows processing methods, bottom layer shows outputs. Feedback loops connect later stages back to earlier stages for validation.

Color Palette: 
- Stage 1 (Data Preprocessing): Azure Blue
- Stage 2 (Fan Vote Estimation): Slate Grey  
- Stage 3 (Voting Comparison): Coral Orange
- Stage 4 (Factor Analysis): Forest Green
- Stage 5 (New System): Deep Purple

**[ZONE 1: LEFT PANEL - DATA PREPROCESSING (Stage 1)]**

Container: Left rectangular panel with rounded corners.

Visual Structure:
- Top section: Icon of a database with label "[Raw DWTS Data]"
- Middle section: Three processing blocks arranged vertically:
  - Block 1: Icon of a cleaning brush with label "[Data Cleaning]"
  - Block 2: Icon of a calculator with label "[Score Calculation]"
  - Block 3: Icon of a checklist with label "[Data Validation]"
- Bottom section: Icon of a processed database with label "[Processed Data]"

Key Text Labels:
- "[Raw DWTS Data]"
- "[Data Cleaning]"
- "[Score Calculation]"
- "[Data Validation]"
- "[Processed Data]"

**[ZONE 2: SECOND PANEL - FAN VOTE ESTIMATION (Stage 2)]**

Container: Second rectangular panel, connected to Zone 1 via arrow.

Visual Structure:
- Top section: Icon of processed data flowing in from left
- Middle section: Central processing unit showing:
  - Left: Mathematical equation icon with label "[Constraint Optimization]"
  - Center: Algorithm icon with label "[Multi-start SLSQP]"
  - Right: Optimization curve icon with label "[Global Optimization]"
- Bottom section: Two output icons:
  - Left: Chart icon with label "[Fan Vote Estimates]"
  - Right: Uncertainty icon with label "[Uncertainty Analysis]"

Key Text Labels:
- "[Constraint Optimization]"
- "[Multi-start SLSQP]"
- "[Global Optimization]"
- "[Fan Vote Estimates]"
- "[Uncertainty Analysis]"
- "[90% Accuracy]"

**[ZONE 3: THIRD PANEL - VOTING METHOD COMPARISON (Stage 3)]**

Container: Third rectangular panel, connected to Zone 2 via arrow.

Visual Structure:
- Top section: Two parallel processing streams:
  - Top stream: Ranking icon with label "[Rank-based Method]"
  - Bottom stream: Percentage icon with label "[Percent-based Method]"
- Middle section: Comparison icon (two overlapping circles) with label "[Method Comparison]"
- Bottom section: Three output icons:
  - Left: Accuracy chart with label "[Accuracy: 60.5% vs 97.0%]"
  - Center: Controversy icon with label "[Controversial Cases]"
  - Right: Agreement icon with label "[37.46% Disagreement]"

Key Text Labels:
- "[Rank-based Method]"
- "[Percent-based Method]"
- "[Method Comparison]"
- "[Accuracy: 60.5% vs 97.0%]"
- "[Controversial Cases]"
- "[37.46% Disagreement]"

**[ZONE 4: FOURTH PANEL - FACTOR IMPACT ANALYSIS (Stage 4)]**

Container: Fourth rectangular panel, connected to Zone 3 via arrow.

Visual Structure:
- Top section: Four factor icons arranged horizontally:
  - Icon 1: Person icon with label "[Professional Dancers]"
  - Icon 2: Age icon with label "[Age]"
  - Icon 3: Industry icon with label "[Industry]"
  - Icon 4: Globe icon with label "[Region]"
- Middle section: Two analysis paths:
  - Top path: Judge icon with label "[Judge Score Impact]"
  - Bottom path: Fan icon with label "[Fan Vote Impact]"
- Bottom section: Statistical chart icon with label "[Differential Impact Analysis]"

Key Text Labels:
- "[Professional Dancers]"
- "[Age]"
- "[Industry]"
- "[Region]"
- "[Judge Score Impact]"
- "[Fan Vote Impact]"
- "[Differential Impact Analysis]"

**[ZONE 5: RIGHT PANEL - NEW VOTING SYSTEM (Stage 5)]**

Container: Right rectangular panel, connected to Zone 4 via arrow.

Visual Structure:
- Top section: Machine learning icon with label "[ML System (LightGBM)]"
- Middle section: Feature engineering icon with label "[12-D Feature Vector]"
- Bottom section: Three output icons:
  - Left: Accuracy icon with label "[97.99% Accuracy]"
  - Center: Fairness icon with label "[Fairness Analysis]"
  - Right: Recommendation icon with label "[System Proposal]"

Key Text Labels:
- "[ML System (LightGBM)]"
- "[12-D Feature Vector]"
- "[97.99% Accuracy]"
- "[Fairness Analysis]"
- "[System Proposal]"

**[CONNECTIONS]**

Forward Data Flow (Left to Right):
- Thick arrow from Zone 1 "[Processed Data]" to Zone 2 input
- Thick arrow from Zone 2 "[Fan Vote Estimates]" to Zone 3 input
- Thick arrow from Zone 3 "[Method Comparison]" to Zone 4 input
- Thick arrow from Zone 4 "[Differential Impact Analysis]" to Zone 5 input

Feedback Loops (Right to Left):
- Curved arrow from Zone 5 "[97.99% Accuracy]" back to Zone 2 "[90% Accuracy]" for validation comparison
- Curved arrow from Zone 4 "[Differential Impact Analysis]" back to Zone 2 "[Constraint Optimization]" for feature enhancement
- Curved arrow from Zone 3 "[Controversial Cases]" back to Zone 2 "[Uncertainty Analysis]" for case-specific validation

Cross-Stage Connections:
- Thin arrow from Zone 1 "[Processed Data]" directly to Zone 4 "[Factor Impact Analysis]" (bypassing stages 2-3 for direct factor analysis)
- Thin arrow from Zone 2 "[Fan Vote Estimates]" directly to Zone 5 "[12-D Feature Vector]" (feature input)

**[ADDITIONAL ELEMENTS]**

Performance Metrics (Top Banner):
- Small badge above Zone 2: "[90.0% Accuracy]"
- Small badge above Zone 3: "[97.0% Best Method]"
- Small badge above Zone 5: "[97.99% Final Accuracy]"

Data Flow Indicators:
- Small arrow icons on all connection lines indicating direction
- Small data packet icons on connection lines showing data type

---

## END PROMPT

### 4. Technical Specifications

**Dimensions:** 16:9 aspect ratio, 4K resolution (3840×2160 pixels)

**Color Scheme:**
- Background: Pure white (#FFFFFF)
- Stage 1 (Azure Blue): #4A90E2
- Stage 2 (Slate Grey): #708090
- Stage 3 (Coral Orange): #FF7F50
- Stage 4 (Forest Green): #228B22
- Stage 5 (Deep Purple): #6A5ACD
- Arrows: #2C3E50 (dark grey)
- Text: #1A1A1A (near black)

**Typography:**
- Main labels: Bold Sans-Serif, 24pt
- Sub-labels: Regular Sans-Serif, 18pt
- Metrics: Bold Sans-Serif, 20pt

**Line Styles:**
- Main connections: 3pt solid lines
- Feedback loops: 2pt dashed lines
- Cross-connections: 1.5pt dotted lines

### 5. Quality Checklist

Before finalizing, ensure:
- [ ] All five stages are clearly visible and labeled
- [ ] Data flow arrows are unidirectional (left to right for main flow)
- [ ] Feedback loops are clearly distinguished from forward flow
- [ ] Performance metrics are prominently displayed
- [ ] Color coding is consistent across all zones
- [ ] No meta-labels appear in the final image
- [ ] All key text labels are readable and properly positioned
- [ ] Overall composition is balanced and professional
- [ ] Diagram tells a clear story of the research pipeline

---

## Usage Instructions

1. Copy the entire prompt above (from "BEGIN PROMPT" to "END PROMPT")
2. Paste into your image generation tool (e.g., DALL-E, Midjourney, Stable Diffusion, or nano banana)
3. Adjust the prompt if needed based on your specific tool's requirements
4. Generate the image
5. Review and refine if necessary

## Expected Output

The generated diagram should show:
- A clear left-to-right pipeline of 5 stages
- Distinct color coding for each stage
- Forward data flow and feedback loops
- Performance metrics displayed prominently
- Professional academic appearance suitable for a competition paper
