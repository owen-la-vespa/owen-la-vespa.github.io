import {useState, useEffect, useRef, useCallback} from 'react';
import {
    Container,
    Row,
    Col,
    Card,
    Form,
    Button,
    Nav,
    Navbar,
    ProgressBar,
    ListGroup,
    Badge,
    Stack,
    Tooltip,
    OverlayTrigger,
    Collapse,
    Modal,
    Alert
} from 'react-bootstrap';
import throttle from 'lodash.throttle';

// --- Helper function for colormapping (Example: Jet) ---
// Input: value (0-1), Output: [r, g, b] array (0-255)
function applyJetColormap(value) {
    const cmap = [
        [0, 0, 0.5625], [0, 0, 0.625], [0, 0, 0.6875], [0, 0, 0.75], [0, 0, 0.8125], [0, 0, 0.875], [0, 0, 0.9375], [0, 0, 1],
        [0, 0.0625, 1], [0, 0.125, 1], [0, 0.1875, 1], [0, 0.25, 1], [0, 0.3125, 1], [0, 0.375, 1], [0, 0.4375, 1], [0, 0.5, 1],
        [0, 0.5625, 1], [0, 0.625, 1], [0, 0.6875, 1], [0, 0.75, 1], [0, 0.8125, 1], [0, 0.875, 1], [0, 0.9375, 1], [0, 1, 1],
        [0.0625, 1, 0.9375], [0.125, 1, 0.875], [0.1875, 1, 0.8125], [0.25, 1, 0.75], [0.3125, 1, 0.6875], [0.375, 1, 0.625], [0.4375, 1, 0.5625], [0.5, 1, 0.5],
        [0.5625, 1, 0.4375], [0.625, 1, 0.375], [0.6875, 1, 0.3125], [0.75, 1, 0.25], [0.8125, 1, 0.1875], [0.875, 1, 0.125], [0.9375, 1, 0.0625], [1, 1, 0],
        [1, 0.9375, 0], [1, 0.875, 0], [1, 0.8125, 0], [1, 0.75, 0], [1, 0.6875, 0], [1, 0.625, 0], [1, 0.5625, 0], [1, 0.5, 0],
        [1, 0.4375, 0], [1, 0.375, 0], [1, 0.3125, 0], [1, 0.25, 0], [1, 0.1875, 0], [1, 0.125, 0], [1, 0.0625, 0], [1, 0, 0],
        [0.9375, 0, 0], [0.875, 0, 0], [0.8125, 0, 0], [0.75, 0, 0], [0.6875, 0, 0], [0.625, 0, 0], [0.5625, 0, 0], [0.5, 0, 0]
    ];
    const n = cmap.length;
    const i = Math.min(n - 1, Math.floor(value * n));
    return cmap[i].map(c => Math.round(c * 255)); // Scale to 0-255
}


const ANALYSIS_DATA = [
    {
        "id": "mobilenetv2_squirrel", // Heatmap 10
        "label": "MobileNetV2 - Squirrel",
        "modelName": "MobileNetV2",
        "originalImageUrl": "/images/squirrel_original.jpg",
        "heatmapOverlayUrl": "/images/squirrel_MobileNetV2_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/squirrel_MobileNetV2_heatmap_color.png",
        "heatmapRawUrl": "/images/squirrel_MobileNetV2_heatmap_raw.png",
        "topPredictions": [{"className": "fox_squirrel", "confidence": 0.5383}, {
            "className": "hamster",
            "confidence": 0.0214
        }, {"className": "macaque", "confidence": 0.0196}],
        "explanation": "MobileNetV2 (the AI model chosen) identified the squirrel in the image with medium confidence, The AI focused evenly in a large circle area on the squirrel's neck, upper body and head area (distinct red area).<br>Interestingly, all the AI models mistook the 'Red Squirrel' for a 'fox squirrel' due to the AI model not having training on 'Red squirrels'. This highlights that tools like Grad-CAM do not always give the whole context for why the AI predicts what it does, however it can aid someone in determining if the AI missed important features in leading up to a misclassification. It seems to have picked the next best thing.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    {
        "id": "resnet50_squirrel", // Heatmap 11
        "label": "ResNet50 - Squirrel",
        "modelName": "ResNet50",
        "originalImageUrl": "/images/squirrel_original.jpg",
        "heatmapOverlayUrl": "/images/squirrel_ResNet50_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/squirrel_ResNet50_heatmap_color.png",
        "heatmapRawUrl": "/images/squirrel_ResNet50_heatmap_raw.png",
        "topPredictions": [{"className": "fox_squirrel", "confidence": 0.9971}, {
            "className": "hare",
            "confidence": 0.0016
        }, {"className": "marmot", "confidence": 0.0003}],
        "explanation": "ResNet50 (the AI model chosen) pinpointed the squirrel in the image with very high confidence, by looking at its bushy tail (large red area) and details around the paws holding food (separate red area).<br>Interestingly, all the AI models mistook the 'Red Squirrel' for a 'fox squirrel' due to the AI model not having training on 'Red squirrels'. This highlights that tools like Grad-CAM do not always give the whole context for why the AI predicts what it does, however it can aid someone in determining if the AI missed important features in leading up to a misclassification. It seems to have picked the next best thing.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    {
        "id": "vgg16_squirrel", // Heatmap 12
        "label": "VGG16 - Squirrel",
        "modelName": "VGG16",
        "originalImageUrl": "/images/squirrel_original.jpg",
        "heatmapOverlayUrl": "/images/squirrel_VGG16_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/squirrel_VGG16_heatmap_color.png",
        "heatmapRawUrl": "/images/squirrel_VGG16_heatmap_raw.png",
        "topPredictions": [{"className": "fox_squirrel", "confidence": 0.9987}, {
            "className": "marmot",
            "confidence": 0.0007
        }, {"className": "hare", "confidence": 0.0002}],
        "explanation": "VGG16 (the AI model chosen) identified the squirrel in the image with very high confidence, by focusing intensely on the squirrel's head region (red area) and less importantly on spots of the body and tail (yellow/green area), largely ignoring most of the body.<br>Interestingly, all the AI models mistook the 'Red Squirrel' for a 'fox squirrel' due to the AI model not having training on 'Red squirrels'. This highlights that tools like Grad-CAM do not always give the whole context for why the AI predicts what it does, however it can aid someone in determining if the AI missed important features in leading up to a misclassification. It seems to have picked the next best thing.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    {
        "id": "efficientnetb0_squirrel", // Heatmap 9
        "label": "EfficientNetB0 - Squirrel",
        "modelName": "EfficientNetB0",
        "originalImageUrl": "/images/squirrel_original.jpg",
        "heatmapOverlayUrl": "/images/squirrel_EfficientNetB0_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/squirrel_EfficientNetB0_heatmap_color.png",
        "heatmapRawUrl": "/images/squirrel_EfficientNetB0_heatmap_raw.png",
        "topPredictions": [{"className": "fox_squirrel", "confidence": 0.9545}, {
            "className": "hare",
            "confidence": 0.003
        }, {"className": "marmot", "confidence": 0.0017}],
        "explanation": "EfficientNetB0 (the AI model chosen) identified the squirrel in the image with very high confidence by looking almost exclusively at the distinct texture and shape of its prominent tail, shown by the intense red highlight.<br>Interestingly, all the AI models mistook the 'Red Squirrel' for a 'fox squirrel' due to the AI model not having training on 'Red squirrels'. This highlights that tools like Grad-CAM do not always give the whole context for why the AI predicts what it does, however it can aid someone in determining if the AI missed important features in leading up to a misclassification. It seems to have picked the next best thing.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    // --- Elephants --- (Heatmaps 1-4)
    {
        "id": "mobilenetv2_elephants", // Heatmap 2
        "label": "MobileNetV2 - Elephants",
        "modelName": "MobileNetV2",
        "originalImageUrl": "/images/elephants_original.jpg",
        "heatmapOverlayUrl": "/images/elephants_MobileNetV2_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/elephants_MobileNetV2_heatmap_color.png",
        "heatmapRawUrl": "/images/elephants_MobileNetV2_heatmap_raw.png",
        "topPredictions": [{"className": "African_elephant", "confidence": 0.5235}, {
            "className": "Indian_elephant",
            "confidence": 0.2633
        }, {"className": "tusker", "confidence": 0.0919}],
        "explanation": "MobileNetV2 (the AI model chosen) correctly identified the image subject as an African Elephant with medium confidence, concentrating (red/yellow) mainly on the adult's distinctive forehead shape and the upper curve of its back/ear region and the baby's face (yellow/green).",
        "focusType": "logical",
        "suspicionReason": null,
        "correctFeatureOptionKey": "B"
    },
    {
        "id": "resnet50_elephants", // Heatmap 3
        "label": "ResNet50 - Elephants",
        "modelName": "ResNet50",
        "originalImageUrl": "/images/elephants_original.jpg",
        "heatmapOverlayUrl": "/images/elephants_ResNet50_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/elephants_ResNet50_heatmap_color.png",
        "heatmapRawUrl": "/images/elephants_ResNet50_heatmap_raw.png",
        "topPredictions": [{"className": "African_elephant", "confidence": 0.3969}, {
            "className": "tusker",
            "confidence": 0.3825
        }, {"className": "Indian_elephant", "confidence": 0.188}],
        "explanation": "Predicting African Elephant with medium/low confidence (with Tusker not far behind it), ResNet50 (the AI model chosen) focused very strongly (intense red) on the baby elephant's ear and the area around its body. It also focused somewhat on the background grass on the right (yellow/green).<br>This indicates the model might have picked up features from the background during training as well which is not ideal when identifying the subject of the picture. This might contribute to lower confidence.",
        "focusType": "suspicious",
        "suspicionReason": "background focus"
    },
    {
        "id": "vgg16_elephants", // Heatmap 4
        "label": "VGG16 - Elephants",
        "modelName": "VGG16",
        "originalImageUrl": "/images/elephants_original.jpg",
        "heatmapOverlayUrl": "/images/elephants_VGG16_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/elephants_VGG16_heatmap_color.png",
        "heatmapRawUrl": "/images/elephants_VGG16_heatmap_raw.png",
        "topPredictions": [{"className": "African_elephant", "confidence": 0.5458}, {
            "className": "tusker",
            "confidence": 0.3091
        }, {"className": "Indian_elephant", "confidence": 0.1434}],
        "explanation": "VGG16 (the AI model chosen) primarily used the area around the adult's tusk and the base of the trunk (red spot) to classify this subject with medium confidence as an African Elephant. It also took features from the baby elephant's ear and back (orange/yellow area). These are all prominent features of the subject which the AI model accurately classified the elephants with.",
        "focusType": "suspicious",
        "suspicionReason": "illogical area"
    },
    {
        "id": "efficientnetb0_elephants", // Heatmap 1
        "label": "EfficientNetB0 - Elephants",
        "modelName": "EfficientNetB0",
        "originalImageUrl": "/images/elephants_original.jpg",
        "heatmapOverlayUrl": "/images/elephants_EfficientNetB0_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/elephants_EfficientNetB0_heatmap_color.png",
        "heatmapRawUrl": "/images/elephants_EfficientNetB0_heatmap_raw.png",
        "topPredictions": [{"className": "tusker", "confidence": 0.3111}, {
            "className": "African_elephant",
            "confidence": 0.2994
        }, {"className": "Indian_elephant", "confidence": 0.2306}],
        "explanation": "Predicting 'tusker' slightly higher than 'African Elephant', EfficientNetB0 (the AI model chosen) seems to have had problems with its focus. Having equally intense focus (red/yellow areas) on the adult's forehead texture and upper trunk (key elephant features), whilst equally intensely focused on the background (intense red), which indicates the model had trouble detecting the elephants. This likely contributed to the misclassification of the subject in the image.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    // --- Puppies --- (Heatmaps 5-8)
    {
        "id": "mobilenetv2_puppies", // Heatmap 6
        "label": "MobileNetV2 - Puppies",
        "modelName": "MobileNetV2",
        "originalImageUrl": "/images/puppies_original.jpg",
        "heatmapOverlayUrl": "/images/puppies_MobileNetV2_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/puppies_MobileNetV2_heatmap_color.png",
        "heatmapRawUrl": "/images/puppies_MobileNetV2_heatmap_raw.png",
        "topPredictions": [{"className": "Labrador_retriever", "confidence": 0.6929}, {
            "className": "golden_retriever",
            "confidence": 0.1265
        }, {"className": "kuvasz", "confidence": 0.0184}],
        "explanation": "MobileNetV2 (the AI model chosen) correctly identified the subjects with high confidence as Labrador Retrievers primarily by focusing (red/yellow) on the head shape and snout area of the puppy on the right. These are important features of the subject likely contributing to the correct prediction.",
        "focusType": "logical",
        "suspicionReason": null,
        "correctFeatureOptionKey": "A"
    },
    {
        "id": "resnet50_puppies", // Heatmap 7
        "label": "ResNet50 - Puppies",
        "modelName": "ResNet50",
        "originalImageUrl": "/images/puppies_original.jpg",
        "heatmapOverlayUrl": "/images/puppies_ResNet50_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/puppies_ResNet50_heatmap_color.png",
        "heatmapRawUrl": "/images/puppies_ResNet50_heatmap_raw.png",
        "topPredictions": [{"className": "Saluki", "confidence": 0.4771}, {
            "className": "golden_retriever",
            "confidence": 0.1973
        }, {"className": "English_setter", "confidence": 0.1321}],
        "explanation": "Interestingly, ResNet50 (the AI model chosen) misclassified this as a Saluki with medium confidence. The model focused its attention intensely (red area) on the region centered between the two puppies', covering parts of both torsos, which likely resembled features associated with the Saluki breed when the AI was trained. The focus as shown on the heatmap seems as though the AI missed important features of the subject (like the head shape and face), this may have contributed to the misclassification. However, it's focus still resides on the subject matter which is a good sign the AI was successful in detecting the subject.",
        "focusType": "suspicious",
        "suspicionReason": "misclassification focus"
    },
    {
        "id": "vgg16_puppies", // Heatmap 8
        "label": "VGG16 - Puppies",
        "modelName": "VGG16",
        "originalImageUrl": "/images/puppies_original.jpg",
        "heatmapOverlayUrl": "/images/puppies_VGG16_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/puppies_VGG16_heatmap_color.png",
        "heatmapRawUrl": "/images/puppies_VGG16_heatmap_raw.png",
        "topPredictions": [{"className": "Labrador_retriever", "confidence": 0.4282}, {
            "className": "golden_retriever",
            "confidence": 0.3985
        }, {"className": "Saluki", "confidence": 0.0554}],
        "explanation": "VGG16 (the AI model chosen) identified the Labrador Retrievers correctly with medium/low confidence. the heatmap highlights the most importance (red heatmap area) on the top of the heads and mouth region of the puppy on the right. It also highlighted parts of the torsos of both puppies with less importance (yellow/green areas). The AI is detecting the important features of the puppies in this image, likely helping it predict the animal correctly.",
        "focusType": "logical",
        "suspicionReason": null,
        "correctFeatureOptionKey": "C"
    },
    {
        "id": "efficientnetb0_puppies", // Heatmap 5
        "label": "EfficientNetB0 - Puppies",
        "modelName": "EfficientNetB0",
        "originalImageUrl": "/images/puppies_original.jpg",
        "heatmapOverlayUrl": "/images/puppies_EfficientNetB0_gradcam_overlay.png",
        "heatmapOnlyUrl": "/images/puppies_EfficientNetB0_heatmap_color.png",
        "heatmapRawUrl": "/images/puppies_EfficientNetB0_heatmap_raw.png",
        "topPredictions": [{"className": "Labrador_retriever", "confidence": 0.4497}, {
            "className": "golden_retriever",
            "confidence": 0.3611
        }, {"className": "kuvasz", "confidence": 0.0344}],
        "explanation": "EfficientNetB0 (the AI model chosen) classified the subject of the image correctly as Labrador Retrievers with medium confidence, focusing on important features of the subject. The red/yellow area surrounds a large circle which includes the heads and upper bodies, with the most intense focus (red area) on the head of the puppy on the left. The AI was able to pick up on an important region of the image containing many important features. The outline of interest on the heatmap shows that the AI recognised a loose outline of the puppies too.",
        "focusType": "logical",
        "suspicionReason": null,
        "correctFeatureOptionKey": "A"
    }
];

// --- 2. Generate Base Examples for Selection UI ---
const EXAMPLE_IMAGES_BASE = ANALYSIS_DATA.reduce((acc, current) => {
    // Extract the base image key (e.g., 'squirrel', 'elephants')
    const imageKey = current.id.split('_').slice(1).join('_'); // Handle keys like 'my_image_key'
    if (!acc.find(item => item.key === imageKey)) {
        acc.push({
            key: imageKey,
            label: imageKey.charAt(0).toUpperCase() + imageKey.slice(1), // Simple capitalization
            src: current.originalImageUrl, // Use the original image URL for the thumbnail
        });
    }
    return acc;
}, []);

// --- 3. Define Available Models (Mapping keys to names/icons) ---
const MODELS_AVAILABLE = {
    mobilenetv2: {
        key: 'mobilenetv2',
        name: 'MobileNetV2',
        description: 'Lightweight', // Keep original short description
        laymanDescription: 'Good at quickly identifying main objects, often used on mobile devices.', // Added layman description
        icon: 'phone'
    },
    resnet50: {
        key: 'resnet50',
        name: 'ResNet50',
        description: 'Deep',
        laymanDescription: 'A powerful model that learns complex details and textures effectively.',
        icon: 'layers'
    },
    vgg16: {
        key: 'vgg16',
        name: 'VGG16',
        description: 'Classic',
        laymanDescription: 'An older but influential model, known for focusing on shapes and textures.',
        icon: 'grid-3x3'
    },
    efficientnetb0: {
        key: 'efficientnetb0',
        name: 'EfficientNetB0',
        description: 'Balanced',
        laymanDescription: 'Designed to be efficient and accurate, often good at recognizing fine-grained details.',
        icon: 'lightning'
    }
};

// --- ADD Mapping from Image Key to its "True" or Intended Label ---
const TRUE_IMAGE_LABELS = {
    'squirrel': 'Red Squirrel',
    'elephants': 'African Elephant(s)', // Be specific if applicable
    'puppies': 'Labrador Retriever Puppies'
};

function getStage1IntroText(example, index) {
    const prediction = example.topPredictions[0]?.className.replace(/_/g, ' ') || 'Unknown';
    const trueLabel = TRUE_IMAGE_LABELS[example.id.split('_').slice(1).join('_')] || 'subject'; // Get true label

    const templates = [
        () => `Alright Detective! 🕵️‍♀️ We need to find out where the AI was focusing on to makes sense of its prediction. Look at the colorful heatmap overlay – where are the <strong class="text-info">brightest</strong> (red/yellow) spots?</p><p>Click the button below that best describes what the AI 'looked' at the most!`,
        (pred, subject) => `Let's investigate this 🔎 '<strong class="text-muted">${subject}</strong>' image! Where did the AI '<strong class="text-info">shine its spotlight</strong>' (the bright red/yellow parts)?</p><p>Pick the button describing those key features.`,
        (pred) => `Hmm 🤔, the AI sees a '<strong class="text-primary">${pred}</strong>' here. Can you figure out what it was looking at, from the <strong class="text-info">bright</strong> heatmap clues (red/yellow areas)?</p><p>Choose the description that fits!`
    ];

    // Cycle through templates based on the example index
    const templateIndex = index % templates.length;
    const selectedTemplate = templates[templateIndex];

    // Call the selected template function with necessary arguments
    // The function signature determines which arguments are used (prediction, trueLabel, etc.)
    // This example primarily uses prediction, but template B uses trueLabel too.
    return selectedTemplate(prediction, trueLabel);
}

// --- Helper to get varied intro text for Stage 2 ---
function getStage2IntroText(example, index) {
    const prediction = example.topPredictions[0]?.className.replace(/_/g, ' ') || 'Unknown';
    const trueLabel = TRUE_IMAGE_LABELS[example.id.split('_').slice(1).join('_')] || 'subject'; // Get true label

    const templates = [
        // Option 1: Direct comparison prompt
        (pred, subject) => `Alright Detective, put it all together! 🧩 The AI predicted <strong class="text-primary">'${pred}'</strong>, the image actually shows a <strong class="text-secondary">'${subject}'</strong>, and the heatmap highlights <strong class="text-info">these areas</strong> (bright spots). Does everything add up (<strong class="text-success">Logical</strong>), or is something fishy (<strong class="text-warning">Suspicious</strong>)?`,

        // Option 2: Focus on potential mismatch
        (pred, subject) => `Time to check the AI's work! It said <strong class="text-primary">'${pred}'</strong> (and it's really a <strong class="text-secondary">'${subject}'</strong>). Does the <strong class="text-info">heatmap focus</strong> (red/yellow) make sense, or does it look out of place? Decide: <strong class="text-success">Logical</strong> or <strong class="text-warning">Suspicious</strong>?`,

        // Option 3: Simple check-in
        (pred, subject) => `Let's double-check! Prediction: <strong class="text-primary">'${pred}'</strong>. Real Subject: <strong class="text-secondary">'${subject}'</strong>. Heatmap Focus: <strong class="text-info">(Bright Areas)</strong>. Does the AI's 'highlighting' seem reasonable for its prediction on this image? <strong class="text-success">Logical</strong> or <strong class="text-warning">Suspicious</strong>?`,

        // Option 4: Questioning the evidence
        (pred, subject) => `Examine the evidence! The AI claims <strong class="text-primary">'${pred}'</strong> for this <strong class="text-secondary">'${subject}'</strong>. Does the <strong class="text-info">heatmap's focus</strong> back up its claim (<strong class="text-success">Logical</strong>), or does the evidence seem weak or misleading (<strong class="text-warning">Suspicious</strong>)?`
    ];

    // Cycle through templates based on the example index
    const templateIndex = index % templates.length;
    const selectedTemplate = templates[templateIndex];

    // Call the selected template function
    return selectedTemplate(prediction, trueLabel);
}

// OPTIONS FOR STAGE 1
const stage1OptionsLookup = {
    'mobilenetv2_elephants': [
        {key: 'A', text: "The adult's forehead texture and shape."},
        {key: 'B', text: "The baby elephant's face and adult's head and back regions"},
        {key: 'C', text: "The grassy area below the elephants."}
    ],
    'mobilenetv2_puppies': [
        {key: 'A', text: "Only the heads of both dogs"},
        {key: 'B', text: "The body of the dog on the right"},
        {key: 'C', text: "The full area of the dogs"}
    ],
    'vgg16_puppies': [
        {key: 'A', text: "Only the heads of both dogs"},
        {key: 'B', text: "The background area on the right of the dogs"},
        {key: 'C', text: "The top of the head and mouth region of the puppy on the right"}
    ],
    'efficientnetb0_puppies': [
        {key: 'A', text: "The heads and upper body of both dogs"},
        {key: 'B', text: "Only the outline of the dogs"},
        {key: 'C', text: "The full area of the dogs"}
    ],
    'default': [ // Fallback
        {key: 'A', text: "Feature A"}, {key: 'B', text: "Feature B"}, {
            key: 'C',
            text: "Feature C"
        }
    ]
};


export default function GradCAMVisualizer() {
    // --- State Management ---
    const [modelPrediction, setModelPrediction] = useState({
        id: null,
        label: '',
        confidence: 0,
        otherPredictions: [], // To store 2nd and 3rd predictions
        explanation: ''
    });
    const [showAdvancedView, setShowAdvancedView] = useState(false); // Start with details shown
    const [selectedImageType, setSelectedImageType] = useState(EXAMPLE_IMAGES_BASE[0]?.key || null);
    const [selectedImageOriginalUrl, setSelectedImageOriginalUrl] = useState(EXAMPLE_IMAGES_BASE[0]?.src || null);
    const [heatmapRawUrl, setHeatmapRawUrl] = useState(null);
    const [heatmapOpacity, setHeatmapOpacity] = useState(0.7);
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [threshold, setThreshold] = useState(0);
    const [selectedModelKey, setSelectedModelKey] = useState(Object.keys(MODELS_AVAILABLE)[0] || null);
    const [openExplanation, setOpenExplanation] = useState(''); // Stores the ID of the open section ('what', 'how', 'using')
    const [analysisStatus, setAnalysisStatus] = useState('idle');
    const [animatedProgress, setAnimatedProgress] = useState({top: 0, others: []}); // Store progress for top and other
    // --- State/Refs for Highlighting ---
    const canvasRef = useRef(null);
    const offscreenCanvasRef = useRef(null);
    // --- Store the actual ImageData object, not just the .data array ---
    const rawHeatmapImageDataRef = useRef(null);
    // --- State to explicitly track when heatmap data IS ready ---
    const [isHeatmapDataReady, setIsHeatmapDataReady] = useState(false);
    const [hoverInfo, setHoverInfo] = useState(null); // { canvasX, canvasY, intensity (0-1), pageX, pageY }
    // Add state for the game
    const [showGameModal, setShowGameModal] = useState(false);
    const [gameStage, setGameStage] = useState(1); // 1 for Explain Focus, 2 for Anomaly Detector
    const [gameExamplesStage1, setGameExamplesStage1] = useState([]); // Examples for Stage 1
    const [gameExamplesStage2, setGameExamplesStage2] = useState([]); // Examples for Stage 2
    const [currentGameIndex, setCurrentGameIndex] = useState(0); // Index within the current stage's examples
    const [gameFeedback, setGameFeedback] = useState('');
    const [gameScoreStage1, setGameScoreStage1] = useState(0);
    const [gameScoreStage2, setGameScoreStage2] = useState(0);
    const [showGameAnswer, setShowGameAnswer] = useState(false);

    // Function to set up and start the game
    const setupGame = () => {
        // --- Filter and Shuffle Examples for Stage 1 (Logical Only) ---
        const logicalExamples = ANALYSIS_DATA.filter(ex => ex.focusType === 'logical' && ex.correctFeatureOptionKey); // Ensure it has the key needed for stage 1
        const shuffledLogical = [...logicalExamples].sort(() => 0.5 - Math.random());
        const numStage1Examples = Math.min(3, shuffledLogical.length); // Max 3 examples for stage 1, or fewer if not available
        setGameExamplesStage1(shuffledLogical.slice(0, numStage1Examples));

        // --- Setup for Stage 2 (Max 1 Squirrel) ---
        const numStage2ExamplesTotal = 2; // Total examples desired for stage 2

        // Separate squirrel and non-squirrel examples
        const allPotentialStage2Examples = [...ANALYSIS_DATA]; // Start with all data
        const squirrelExamplesStage2 = allPotentialStage2Examples.filter(ex => ex.id.includes('_squirrel'));
        const nonSquirrelExamplesStage2 = allPotentialStage2Examples.filter(ex => !ex.id.includes('_squirrel'));

        // Shuffle both lists independently
        squirrelExamplesStage2.sort(() => 0.5 - Math.random());
        nonSquirrelExamplesStage2.sort(() => 0.5 - Math.random());

        let selectedStage2Examples = [];
        let includeSquirrel = squirrelExamplesStage2.length > 0 && Math.random() > 0.3; // ~70% chance to include a squirrel if available

        let squirrelToAdd = null;
        if (includeSquirrel) {
            squirrelToAdd = squirrelExamplesStage2[0]; // Take the first (random) squirrel
        }

        // Determine how many non-squirrels needed
        const neededNonSquirrels = numStage2ExamplesTotal - (squirrelToAdd ? 1 : 0);

        // Add non-squirrels (up to the needed amount, or however many are available)
        selectedStage2Examples = nonSquirrelExamplesStage2.slice(0, neededNonSquirrels);

        // Add the squirrel if selected
        if (squirrelToAdd) {
            selectedStage2Examples.push(squirrelToAdd);
        }

        // Final shuffle of the combined stage 2 list
        selectedStage2Examples.sort(() => 0.5 - Math.random());

        // Handle cases where we couldn't gather enough examples total
        if (selectedStage2Examples.length < 1) {
            console.warn("Could not select enough examples for Stage 2 of the game.");
            // Optionally show an alert or disable the game button
        }

        setGameExamplesStage2(selectedStage2Examples);

        // --- Reset Game State ---
        setGameStage(1);
        setCurrentGameIndex(0);
        setGameFeedback('');
        setGameScoreStage1(0);
        setGameScoreStage2(0);
        setShowGameAnswer(false);
        setShowGameModal(true); // Show the modal
    };

    // --- Function to handle answer selection for STAGE 1 ---
    const handleGameStage1Answer = (chosenFeatureExplanationKey) => { // Renamed param for clarity
        if (showGameAnswer) return;
        if (!gameExamplesStage1 || currentGameIndex >= gameExamplesStage1.length) return;

        const currentExample = gameExamplesStage1[currentGameIndex];
        const correctAnswerKey = currentExample.correctFeatureOptionKey; // Get correct key from data
        const isCorrect = chosenFeatureExplanationKey === correctAnswerKey;
        let feedback = '';

        // --- Find the text for the correct answer ---
        const currentOptions = stage1OptionsLookup[currentExample.id] || stage1OptionsLookup['default'];
        const correctOptionObject = currentOptions.find(opt => opt.key === correctAnswerKey);
        const correctOptionText = correctOptionObject ? `"${correctOptionObject.text}"` : "the correct features"; // Get text or fallback
        // Find text for the chosen answer (for incorrect feedback)
        const chosenOptionObject = currentOptions.find(opt => opt.key === chosenFeatureExplanationKey);
        const chosenOptionText = chosenOptionObject ? `"${chosenOptionObject.text}"` : "the chosen features";
        // --- End Text Finding ---

        if (isCorrect) {
            setGameScoreStage1(prev => prev + 1);
            // --- Use correctOptionText in feedback ---
            feedback = `Correct! The AI strongly focuses on features described by: ${correctOptionText}, Good eye!.`;
        } else {
            // --- Use correctOptionText and chosenOptionText in feedback ---
            feedback = `Not quite. You chose ${chosenOptionText}, while the AI's main focus (red/yellow area) actually corresponds more closely to: ${correctOptionText}. Look again at where the brightest colors are.`;
        }
        setGameFeedback(feedback);
        setShowGameAnswer(true);
    };

    // --- Function to handle answer selection for STAGE 2 ---

    const handleGameStage2Answer = (userJudgment) => { // userJudgment = 'logical' or 'suspicious'
        if (showGameAnswer) return;
        if (!gameExamplesStage2 || currentGameIndex >= gameExamplesStage2.length) {
            return;
        }

        const currentExample = gameExamplesStage2[currentGameIndex];


        const predictedClass = currentExample.topPredictions[0]?.className.replace(/_/g, ' ') || 'Unknown';
        const imageKey = currentExample.id.split('_').slice(1).join('_');
        const trueLabel = TRUE_IMAGE_LABELS[imageKey] || 'the image subject';

        const correctAnswerType = currentExample.focusType; // Should be 'logical' or 'suspicious'
        const isCorrect = userJudgment === correctAnswerType;

        let feedback = '';

        if (isCorrect) {
            // +++ DEBUG LOGS +++
            console.log("Feedback Path: Correct");
            setGameScoreStage2(prev => prev + 1);
            feedback = `Correct! `;
            if (correctAnswerType === 'logical') {
                console.log("Feedback Detail: Logical was correct"); // +++
                feedback += `This focus on relevant features logically supports the '${predictedClass}' prediction. Good analysis!`;
            } else { // Correctly identified as suspicious
                console.log("Feedback Detail: Suspicious was correct"); // +++
                const reasonText = currentExample.suspicionReason || 'the highlighted area seems irrelevant or illogical';
                if (currentExample.suspicionReason === 'misclassification focus') {
                    feedback += `The AI's prediction of '${predictedClass}' indeed looks suspicious given the image shows a ${trueLabel}. Good eye!`;
                } else {
                    feedback += `This focus is indeed suspicious for predicting '${predictedClass}'. Reason: ${reasonText}. Good eye!`;
                }
            }
        } else { // User was Incorrect
            feedback = `Not quite. `;
            if (correctAnswerType === 'logical') {
                console.log("Feedback Detail: Incorrect, should have been Logical"); // +++
                feedback += `While sometimes heatmaps look odd, this focus on relevant features *is* considered logical support for the '${predictedClass}' prediction.`;
            } else { // User thought it was logical, but it was suspicious
                console.log("Feedback Detail: Incorrect, should have been Suspicious"); // +++
                const reasonText = currentExample.suspicionReason || 'the highlighted area seems irrelevant or illogical';
                if (currentExample.suspicionReason === 'misclassification focus') {
                    feedback += `Although the AI predicted '${predictedClass}', this focus is actually suspicious because it highlights features more relevant to a different class, especially since the image is a ${trueLabel}.`;
                } else {
                    feedback += `This focus is considered suspicious for the '${predictedClass}' prediction. Reason: ${reasonText}. The highlighted areas aren't the best indicators for this class.`;
                }
            }
        }

        setGameFeedback(feedback);
        setShowGameAnswer(true);
    };


    // --- Function to advance the game ---
    const handleNextGameStep = () => {
        setShowGameAnswer(false);
        setGameFeedback('');

        if (gameStage === 1) {
            if (currentGameIndex < gameExamplesStage1.length - 1) {
                setCurrentGameIndex(prev => prev + 1); // Next example in Stage 1
            } else {
                // Finished Stage 1, move to Stage 2
                setGameStage(2);
                setCurrentGameIndex(0); // Reset index for Stage 2
                if (gameExamplesStage2.length === 0) { // Check if stage 2 has examples
                    alert(`Stage 1 Score: ${gameScoreStage1}/${gameExamplesStage1.length}\nNo suspicious examples configured for Stage 2. Game Over!`);
                    handleCloseGameModal();
                }
            }
        } else { // gameStage === 2
            if (currentGameIndex < gameExamplesStage2.length - 1) {
                setCurrentGameIndex(prev => prev + 1); // Next example in Stage 2
            } else {
                // Finished Stage 2, Game Over
                alert(`Game Over!\nStage 1 Score: ${gameScoreStage1}/${gameExamplesStage1.length}\nStage 2 Score: ${gameScoreStage2}/${gameExamplesStage2.length}`);
                handleCloseGameModal();
            }
        }
    };

    // Close modal handler
    const handleCloseGameModal = () => {
        setShowGameModal(false);
        setGameExamples([]);
        setCurrentGameIndex(0);
        setGameFeedback('');
        setShowGameAnswer(false);
    };

    // --- Core Logic Functions ---
    const findAnalysisData = (imageType, modelKey) => {
        if (!imageType || !modelKey) return null;
        const targetId = `${modelKey}_${imageType}`;
        return ANALYSIS_DATA.find(item => item.id === targetId);
    };

    const selectExampleImage = (exampleKey, exampleSrc) => {
        console.log("Selecting example:", exampleKey);
        setSelectedImageType(exampleKey);
        setSelectedImageOriginalUrl(exampleSrc);
        runAnalysis(exampleKey, selectedModelKey);
    };

    const handleModelSelect = (modelKey) => {
        console.log("Selecting model:", modelKey);
        setSelectedModelKey(modelKey);
        if (selectedImageType) {
            runAnalysis(selectedImageType, modelKey);
        }
    };

    const runAnalysis = (imageType, modelKey) => {
        console.log(`Running analysis for Image: ${imageType}, Model: ${modelKey}`);
        if (!imageType || !modelKey) {
            setAnalysisStatus('idle');
            return;
        }
        setAnalysisStatus('loading');
        setHeatmapRawUrl(null);
        setIsHeatmapDataReady(false);
        setAnimatedProgress({top: 0, others: []}); // <<< RESET animated progress on new analysis

        setTimeout(() => {
            const resultData = findAnalysisData(imageType, modelKey);
            if (resultData && resultData.topPredictions && resultData.topPredictions.length > 0) { // Check if topPredictions exist
                console.log("Found analysis data:", resultData);

                // Extract top prediction
                const topPred = resultData.topPredictions[0];

                // Extract other predictions (2nd and 3rd)
                const otherPreds = resultData.topPredictions.slice(1).map(p => ({
                    label: p.className.replace(/_/g, ' '),
                    confidence: Math.round(p.confidence * 100) // Convert to percentage
                }));

                // Update state
                setModelPrediction({
                    id: resultData.id,
                    label: topPred.className.replace(/_/g, ' '),
                    confidence: Math.round(topPred.confidence * 100), // Target confidence %
                    otherPredictions: otherPreds.map(p => ({ // Target other confidences %
                        label: p.label,
                        confidence: Math.round(p.confidence)
                    })),
                    explanation: resultData.explanation
                });

                if (!selectedImageOriginalUrl || !selectedImageOriginalUrl.includes(imageType)) {
                    setSelectedImageOriginalUrl(resultData.originalImageUrl);
                }
                setHeatmapRawUrl(resultData.heatmapRawUrl);
                setAnalysisStatus('complete');
            } else {
                // Handle missing data or empty predictions
                console.error(`Analysis data or topPredictions not found/empty for id: ${modelKey}_${imageType}`);
                setModelPrediction({
                    id: null,
                    label: 'Error',
                    confidence: 0,
                    otherPredictions: [],
                    explanation: 'Analysis data not available or invalid.'
                });
                setHeatmapRawUrl(null);
                setAnalysisStatus('error');
            }
        }, 400);
    };

    // --- Handler for the main toggle ---
    const toggleAdvancedView = () => {
        // Use the functional form of setState to get the *next* state value
        setShowAdvancedView(prevShowAdvanced => {
            const nextShowAdvanced = !prevShowAdvanced; // Determine the next view state

            // Set threshold based on the view we are *switching to*
            if (nextShowAdvanced) {
                // Switching TO Interface 2
                setThreshold(0.2); // Set threshold to 0.2 for Detailed
                console.log("Switching to Interface 2, setting threshold to 0.2");
            } else {
                // Switching TO Interface 1
                setThreshold(0);   // Set threshold to 0 for Simple
                console.log("Switching to Interface 1, setting threshold to 0");
            }

            return nextShowAdvanced; // Return the new state value for setShowAdvancedView
        });
    };

    // --- Effect to Load Raw Heatmap Data ---
    useEffect(() => {
        // Reset readiness flag when URL changes
        setIsHeatmapDataReady(false);
        rawHeatmapImageDataRef.current = null;

        if (heatmapRawUrl) {
            console.log("Loading new raw heatmap image for pixel data:", heatmapRawUrl);
            const heatmapImg = new Image();
            heatmapImg.crossOrigin = "Anonymous";
            heatmapImg.onload = () => {
                if (!offscreenCanvasRef.current) {
                    offscreenCanvasRef.current = document.createElement('canvas');
                }
                const canvas = offscreenCanvasRef.current;
                const ctx = canvas.getContext('2d', {willReadFrequently: true});
                // --- Explicitly set canvas size to match loaded heatmap ---
                canvas.width = heatmapImg.naturalWidth;
                canvas.height = heatmapImg.naturalHeight;
                ctx.drawImage(heatmapImg, 0, 0);
                try {
                    // --- Store the entire ImageData object ---
                    rawHeatmapImageDataRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    console.log("Raw heatmap ImageData loaded. Dimensions:", canvas.width, "x", canvas.height);
                    // --- Set readiness flag ---
                    setIsHeatmapDataReady(true);
                } catch (error) {
                    console.error("Error getting pixel data from raw heatmap:", error);
                    rawHeatmapImageDataRef.current = null;
                    setIsHeatmapDataReady(false);
                }
            };
            heatmapImg.onerror = () => {
                console.error("Failed to load raw heatmap image for pixel data:", heatmapRawUrl);
                rawHeatmapImageDataRef.current = null;
                setIsHeatmapDataReady(false);
            };
            heatmapImg.src = heatmapRawUrl;
        }
    }, [heatmapRawUrl]); // Rerun only when the raw heatmap URL changes

    // --- Canvas Drawing Effect (Checks for data readiness) ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d', {willReadFrequently: true});

        if (!selectedImageOriginalUrl) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }

        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const drawWidth = img.naturalWidth;
            const drawHeight = img.naturalHeight;
            canvas.width = drawWidth; // Set canvas buffer size = original image size
            canvas.height = drawHeight;

            // Draw original image
            ctx.drawImage(img, 0, 0, drawWidth, drawHeight);

            // --- Check heatmap readiness BEFORE proceeding ---
            if (!showHeatmap || !isHeatmapDataReady || !rawHeatmapImageDataRef.current) {
                // If heatmap off, not ready, or data missing, stop after drawing original
                if (showHeatmap && heatmapRawUrl && !isHeatmapDataReady) console.log("Drawing original only - heatmap data not ready yet.");
                return;
            }

            // --- Now we know raw heatmap data is loaded ---
            const loadedHeatmapImageData = rawHeatmapImageDataRef.current;
            const heatmapWidth = loadedHeatmapImageData.width;
            const heatmapHeight = loadedHeatmapImageData.height;

            // --- Check Dimension Mismatch ---
            if (drawWidth !== heatmapWidth || drawHeight !== heatmapHeight) {
                console.error(`Dimension Mismatch! Original: ${drawWidth}x${drawHeight}, Heatmap: ${heatmapWidth}x${heatmapHeight}. Heatmap overlay might be incorrect. Ensure Colab saves raw heatmap at original size.`);
                // Optional: Could attempt resizing heatmap data here, but it's complex and slow.
                // Best fix is ensuring Colab script saves correctly sized raw heatmap.
                // For now, we proceed, but the mapping will be wrong.
            }

            try {
                const originalImageData = ctx.getImageData(0, 0, drawWidth, drawHeight);
                const originalPixels = originalImageData.data;
                const heatmapPixels = loadedHeatmapImageData.data; // Use data from stored ImageData

                // --- Pixel Manipulation Loop ---
                // This loop assumes dimensions match. If they don't, indices will be wrong.
                for (let i = 0; i < originalPixels.length; i += 4) {
                    // Heatmap intensity from Red channel
                    const heatmapIntensity = heatmapPixels[i] / 255.0;

                    if (heatmapIntensity >= threshold) {
                        const [hr, hg, hb] = applyJetColormap(heatmapIntensity);
                        const base_r = originalPixels[i];
                        const base_g = originalPixels[i + 1];
                        const base_b = originalPixels[i + 2];

                        originalPixels[i] = Math.round(hr * heatmapOpacity + base_r * (1 - heatmapOpacity));
                        originalPixels[i + 1] = Math.round(hg * heatmapOpacity + base_g * (1 - heatmapOpacity));
                        originalPixels[i + 2] = Math.round(hb * heatmapOpacity + base_b * (1 - heatmapOpacity));
                    }
                }
                ctx.putImageData(originalImageData, 0, 0); // Put modified data back

                // --- Draw Highlight (if hovering AND in Advanced View) ---
                if (hoverInfo && hoverInfo.intensity >= threshold && showAdvancedView) { // <<< ADDED showAdvancedView condition
                    ctx.beginPath();
                    ctx.arc(hoverInfo.canvasX, hoverInfo.canvasY, 10, 0, 2 * Math.PI, false); // Draw circle radius 10
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)'; // Semi-transparent white fill
                    ctx.fill();
                    ctx.lineWidth = 1.5;
                    ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)'; // Bright yellow border
                    ctx.stroke();
                }

            } catch (error) {
                console.error("Error processing canvas pixel data in drawing effect:", error);
                ctx.drawImage(img, 0, 0, drawWidth, drawHeight); // Fallback
            }
        };
        img.onerror = () => {
            console.error("Failed to load original image:", selectedImageOriginalUrl);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };
        img.src = selectedImageOriginalUrl;

        // Re-run drawing when heatmap data becomes ready OR other states change
    }, [selectedImageOriginalUrl, isHeatmapDataReady, heatmapRawUrl, showHeatmap, heatmapOpacity, threshold, hoverInfo]); // Dependencies

    // --- Easing Function (Cubic Ease Out) ---
    // Input: t (time ratio, 0 to 1)
    // Output: eased ratio (0 to 1)
    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // --- NEW: useEffect for Progress Bar Animation ---
    const animationDuration = 500; // 0.5 seconds in milliseconds
    const animationFrameRef = useRef(); // To store animation frame ID

    useEffect(() => {
        // Only run animation when analysis is complete and we have a target confidence
        if (analysisStatus === 'complete' && modelPrediction.confidence >= 0) {
            let startTime = null;

            const animate = (timestamp) => {
                if (!startTime) startTime = timestamp;
                const elapsedTime = timestamp - startTime;
                const linearProgressRatio = Math.min(elapsedTime / animationDuration, 1);

                // --- Apply Easing Function ---
                const easedRatio = easeOutCubic(linearProgressRatio);
                // --- End Easing ---

                // Calculate current animated values using the eased ratio
                const currentTopProgress = Math.round(modelPrediction.confidence * easedRatio);
                const currentOtherProgress = (modelPrediction.otherPredictions || []).map(pred =>
                    Math.round(pred.confidence * easedRatio) // Apply easing here too
                );

                setAnimatedProgress({top: currentTopProgress, others: currentOtherProgress});

                // Continue animation if linear progress is not complete
                if (linearProgressRatio < 1) {
                    animationFrameRef.current = requestAnimationFrame(animate);
                } else {
                    // Ensure final state is exactly the target
                    setAnimatedProgress({
                        top: modelPrediction.confidence,
                        others: (modelPrediction.otherPredictions || []).map(p => p.confidence)
                    });
                }
            };

            // Start the animation
            animationFrameRef.current = requestAnimationFrame(animate);

            // Cleanup function
            return () => {
                if (animationFrameRef.current) {
                    cancelAnimationFrame(animationFrameRef.current);
                }
            };
        } else {
            // If not complete or no confidence, reset animation progress immediately
            setAnimatedProgress({top: 0, others: []});
        }
        // Trigger when analysis status changes or the target confidence/predictions change
    }, [analysisStatus, modelPrediction.id]); // Depend on modelPrediction.id to re-trigger if the result changes

    // --- Throttled Mouse Move Handler (Updated to check data readiness and use correct dimensions) ---
    const handleMouseMove = useCallback(throttle((event) => {
        // --- Check readiness first ---
        if (!isHeatmapDataReady || !rawHeatmapImageDataRef.current || analysisStatus !== 'complete') {
            setHoverInfo(null);
            return;
        }

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();

        // Get heatmap dimensions from stored ImageData object
        const heatmapWidth = rawHeatmapImageDataRef.current.width;
        const heatmapHeight = rawHeatmapImageDataRef.current.height;

        // Calculate mouse position relative to the *displayed* canvas element size
        const displayScaleX = heatmapWidth / canvas.clientWidth; // heatmap data width / CSS width
        const displayScaleY = heatmapHeight / canvas.clientHeight;// heatmap data height / CSS height
        // Calculate coordinates directly within the heatmap data's coordinate system
        const heatmapX = Math.round((event.clientX - rect.left) * displayScaleX);
        const heatmapY = Math.round((event.clientY - rect.top) * displayScaleY);

        // Calculate canvas buffer coordinates (needed for drawing highlight circle)
        // Usually canvas buffer = heatmap data size after img.onload sets canvas.width/height
        const canvasX = heatmapX;
        const canvasY = heatmapY;


        // Ensure coordinates are within the heatmap data bounds
        if (heatmapX < 0 || heatmapX >= heatmapWidth || heatmapY < 0 || heatmapY >= heatmapHeight) {
            setHoverInfo(null);
            return;
        }

        // Calculate index in the 1D pixel data array
        const heatmapPixelIndex = (heatmapY * heatmapWidth + heatmapX) * 4;

        let intensity = 0;
        const heatmapPixels = rawHeatmapImageDataRef.current.data; // Use .data here
        if (heatmapPixelIndex >= 0 && heatmapPixelIndex < heatmapPixels.length) {
            intensity = heatmapPixels[heatmapPixelIndex] / 255.0;
        } else {
            console.warn("Calculated heatmap index out of bounds:", heatmapPixelIndex);
            setHoverInfo(null);
            return;
        }

        // Update hover state
        setHoverInfo({
            canvasX: canvasX, // Coordinate within the canvas buffer (same as heatmapX here)
            canvasY: canvasY, // Coordinate within the canvas buffer (same as heatmapY here)
            intensity: intensity,
            pageX: event.pageX,
            pageY: event.pageY
        });
    }, 100), [analysisStatus, isHeatmapDataReady]); // Added isHeatmapDataReady dependency


    // --- Mouse Leave Handler ---
    const handleMouseLeave = () => {
        setHoverInfo(null);
    };

    const toggleExplanationSection = (sectionId) => {
        setOpenExplanation(prevOpen => (prevOpen === sectionId ? '' : sectionId));
    };

    // --- Helper for SVG icons ---
    const IconComponent = ({icon}) => {
        const iconMap = {
            'phone': (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor"
                           className="bi bi-phone" viewBox="0 0 16 16">
                <path
                    d="M11 1a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1zM5 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2z"/>
                <path d="M8 14a1 1 0 1 0 0-2 1 1 0 0 0 0 2"/>
            </svg>),
            'layers': (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor"
                            className="bi bi-layers" viewBox="0 0 16 16">
                <path
                    d="M8.235 1.559a.5.5 0 0 0-.47 0l-7.5 4a.5.5 0 0 0 0 .882L3.188 8 .765 9.559a.5.5 0 0 0 0 .882l7.5 4a.5.5 0 0 0 .47 0l7.5-4a.5.5 0 0 0 0-.882L12.813 8l2.922-1.559a.5.5 0 0 0 0-.882zm3.515 7.008L14.438 10 8 13.433 1.562 10 4.25 8.567l3.515 1.874a.5.5 0 0 0 .47 0zM8 9.433 1.562 6 8 2.567 14.438 6z"/>
            </svg>),
            'grid-3x3': (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor"
                              className="bi bi-grid-3x3" viewBox="0 0 16 16">
                <path
                    d="M0 1.5A1.5 1.5 0 0 1 1.5 0h13A1.5 1.5 0 0 1 16 1.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 0 14.5zM1.5 1a.5.5 0 0 0-.5.5V5h4V1zM5 6H1v4h4zm1 4h4V6H6zm4-1V1H6v4zm1-4V1h3.5a.5.5 0 0 1 .5.5V5zm0 1h4v4h-4zm0 5v4h4v-4zm-5 0v4h4v-4zm-5 0v4h4v-4z"/>
            </svg>),
            'lightning': (<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor"
                               className="bi bi-lightning" viewBox="0 0 16 16">
                <path
                    d="M5.52.359A.5.5 0 0 1 6 0h4a.5.5 0 0 1 .424.765L8.926 3.488l1.599.67A.5.5 0 0 1 10.5 5.5v5a.5.5 0 0 1-.376.484L7.5 12.5v2a.5.5 0 0 1-.5.5h-1a.5.5 0 0 1-.5-.5v-2l-2.624-1.016A.5.5 0 0 1 2.5 10.5v-5a.5.5 0 0 1 .337-.474l2.543-.847L3.89.76A.504.504 0 0 1 4.125.264zm1.83 1.496L5.207 5.095a.5.5 0 0 1-.375.164l-2.5.833v3.546l2.125.875v-2.42l.407-.171L8.5 9.961V3.332L7.35 1.855z"/>
            </svg>),
            'image': (<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor"
                           className="bi bi-image text-secondary mb-2" viewBox="0 0 16 16">
                <path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0"/>
                <path
                    d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1z"/>
            </svg>)
        };
        return iconMap[icon] || null;
    };

    // --- JSX Structure ---
    return (
        <div className="bg-light min-vh-100">
            {/* Navigation */}
            <Navbar bg="primary" variant="dark" expand="lg">
                <Container> <Navbar.Brand href="#">Grad-CAM Explanation</Navbar.Brand> </Container>
            </Navbar>

            <Container className="py-4">
                {/* Header */}
                <Row className="mb-4">
                    <Col lg={9} className="mx-auto text-center">
                        <div className="d-flex justify-content-center align-items-center mb-2">
                            <h1 className="display-5 fw-bold text-primary mb-0">Grad-CAM Explanation Tool</h1>
                            {/* --- Toggle Button --- */}
                            <Button
                                variant="outline-secondary"
                                size="sm"
                                className="ms-3"
                                onClick={toggleAdvancedView}
                                title={showAdvancedView ? "Switch to Interface 1" : "Switch to Interface 2"}
                            >
                                {showAdvancedView ? "Switch to Interface 1" : "Switch to Interface 2"}
                            </Button>
                        </div>
                        <p className="lead text-muted"> Explore pre-computed Grad-CAM examples </p>
                    </Col>
                </Row>

                <Row>
                    {/* ==================================== */}
                    {/* == Main Content Column (lg={10}) == */}
                    {/* ==================================== */}
                    <Col lg={9} className="mb-4">
                        <Card className="shadow-sm">
                            <Card.Header className="bg-primary text-white">
                                <h4 className="mb-0">Image Analysis</h4>
                            </Card.Header>
                            <Card.Body>

                                {/* --- Row for Image + Results/Controls Side-by-Side --- */}
                                <Row className="mb-4 align-items-stretch">
                                    {/* --- Column 1: Image Visualization Area --- */}
                                    <Col md={6} className="d-flex flex-column">
                                        {(analysisStatus !== 'idle' || selectedImageType) && selectedImageOriginalUrl && ( // Show if an image is selected/loading
                                            <p className="text-center mb-2 fw-bold"
                                               style={{fontSize: '1rem'}}>
                                                Displaying: <span
                                                className="text-primary">{TRUE_IMAGE_LABELS[selectedImageType] || selectedImageType}</span>
                                            </p>
                                        )}
                                        <div
                                            className="position-relative text-center mx-auto" // Added mx-auto
                                            style={{
                                                width: '100%', // Take full column width
                                                maxWidth: '450px', // Max size like before
                                                minHeight: '200px',
                                                border: '1px solid #dee2e6', // Border can stay if desired
                                                borderRadius: '0.25rem'
                                            }}
                                        >
                                            <canvas
                                                ref={canvasRef}
                                                className={`img-fluid rounded border ${analysisStatus === 'loading' ? 'opacity-50' : ''}`}
                                                style={{
                                                    maxHeight: '450px',
                                                    backgroundColor: '#f8f9fa',
                                                    width: '100%',
                                                    cursor: showAdvancedView ? 'crosshair' : 'default' // Use crosshair only in advanced view
                                                }} // Add crosshair
                                                width={400} height={400}
                                                onMouseMove={handleMouseMove}
                                                onMouseLeave={handleMouseLeave}
                                            />
                                            {analysisStatus === 'loading' && (
                                                <div className="position-absolute top-50 start-50 translate-middle">
                                                    <div className="spinner-border text-primary" role="status"
                                                         style={{width: '3rem', height: '3rem'}}>
                                                        <span className="visually-hidden">Loading...</span>
                                                    </div>
                                                </div>
                                            )}
                                            {!selectedImageOriginalUrl && analysisStatus !== 'loading' && (
                                                <div
                                                    className="position-absolute top-50 start-50 translate-middle text-muted">
                                                    Select an example image.
                                                </div>
                                            )}
                                        </div>
                                        {showAdvancedView ? (
                                            <>
                                                {selectedImageOriginalUrl && (
                                                    <div
                                                        className="text-center mt-1"> {/* Use text-center for alignment */}
                                                        <small
                                                            className="text-muted fst-italic"> {/* Muted and italic */}
                                                            Hover over image for heatmap intensity details.
                                                        </small>
                                                    </div>
                                                )}
                                                <div className="mt-3 border-top pt-3">
                                                    <Form.Label>Select Example Image</Form.Label>
                                                    <div className="d-flex justify-content-center overflow-auto pb-2">
                                                        {EXAMPLE_IMAGES_BASE.map(example => (
                                                            <div key={example.key}
                                                                 className="text-center mx-1 flex-shrink-0">
                                                                <img
                                                                    src={example.src}
                                                                    alt={example.label}
                                                                    className="rounded"
                                                                    style={{
                                                                        width: '80px',
                                                                        height: '80px',
                                                                        cursor: 'pointer',
                                                                        objectFit: 'cover',
                                                                        border: selectedImageType === example.key ? '3px solid #0d6efd' : '1px solid #dee2e6'
                                                                    }}
                                                                    onClick={() => selectExampleImage(example.key, example.src)}
                                                                    loading="lazy"
                                                                />
                                                                <div className="mt-1 text-primary"
                                                                     style={{fontSize: '0.8rem'}}>{example.label}</div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            </>
                                        ) : (<></>)}
                                    </Col>

                                    {/* --- Column 2: Results and Heatmap Controls Area --- */}
                                    <Col md={6} className="d-flex flex-column">
                                        <Stack gap={3} className="h-100">
                                            {/* Analysis Results */}
                                            <div>
                                                {showAdvancedView ? (
                                                    <>
                                                        <h5>Model's Decision & Reasoning</h5>
                                                        {/* --- Render Progress Bars using animatedProgress state --- */}
                                                        {(analysisStatus === 'complete' || analysisStatus === 'loading') && modelPrediction.label && (
                                                            <div>
                                                                {/* Top Prediction Display */}
                                                                <div className="mb-3">
                                                                    <h6>Top Prediction: <span
                                                                        className="fw-bold">{modelPrediction.label || '...'}</span>
                                                                    </h6>
                                                                    <ProgressBar
                                                                        now={animatedProgress.top} // Starts at 0 during load
                                                                        label={`${animatedProgress.top}%`}
                                                                        variant="success"
                                                                        style={{height: '25px'}}
                                                                    />
                                                                </div>

                                                                {/* Display Other Predictions */}
                                                                {modelPrediction.otherPredictions && modelPrediction.otherPredictions.length > 0 && (
                                                                    <div className="mb-3">
                                                                        <h6>Other Likely Predictions:</h6>
                                                                        {modelPrediction.otherPredictions.map((pred, idx) => (
                                                                            <div key={idx}
                                                                                 className="d-flex align-items-center mb-1">
                                                                                <ProgressBar
                                                                                    now={animatedProgress.others[idx] || 0} // Starts at 0 during load
                                                                                    variant="info"
                                                                                    className="flex-grow-1 me-2"
                                                                                    style={{height: '15px'}}
                                                                                />
                                                                                <span
                                                                                    style={{fontSize: '0.85rem'}}>{pred.label} ({animatedProgress.others[idx] || 0}%)</span>
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                                {/* Explanation (only show explanation text when complete) */}
                                                                {analysisStatus === 'complete' && (
                                                                    <div className="bg-light p-2 rounded mt-3 border">
                                                                        {/* --- Inner Div for Animation (uses new class) --- */}
                                                                        <div
                                                                            key={modelPrediction.id || 'no-exp'} // Key triggers re-mount
                                                                            className="explanation-text-container explanation-animate-grow" // Use new animation class
                                                                        >
                                                                            <p className="text-dark mb-0"
                                                                               style={{fontSize: '0.9rem'}}> {/* Changed text-muted to text-dark */}
                                                                                <strong>Explanation:</strong> <span
                                                                                    dangerouslySetInnerHTML={{__html: modelPrediction.explanation}}></span>
                                                                            </p>
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>

                                                        )}
                                                        {/* --- This handles initial idle/error states --- */}
                                                        {(analysisStatus === 'idle' || analysisStatus === 'error') && !modelPrediction.label && (
                                                            <div className="text-center py-4">
                                                                <p className="text-muted">{modelPrediction.explanation || 'Select an image and model.'}</p>
                                                            </div>
                                                        )}
                                                    </>
                                                ) : (
                                                    <>
                                                        {(analysisStatus === 'complete' || analysisStatus === 'loading') && modelPrediction.label && (
                                                            <div>
                                                                {/* Top Prediction Display */}
                                                                <div className="mb-3">
                                                                    <h6>Top Prediction: <span
                                                                        className="fw-bold">{modelPrediction.label || '...'} ({modelPrediction.confidence}%)</span>
                                                                    </h6>
                                                                </div>

                                                                {/* Display Other Predictions */}
                                                                {modelPrediction.otherPredictions && modelPrediction.otherPredictions.length > 0 && (
                                                                    <div className="mb-3 justify-content-center">
                                                                        <h6>Other Likely Predictions:</h6>
                                                                        {modelPrediction.otherPredictions.map((pred, idx) => (
                                                                            <div key={idx}
                                                                                 className="d-flex align-items-center mb-1">
                                                                                 <span style={{fontSize: '0.85rem'}}>
                                                                            {idx + 2}. {pred.label} ({pred.confidence}%)
                                                                        </span>
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                )}

                                                            </div>

                                                        )}
                                                        {/* --- This handles initial idle/error states --- */}
                                                        {(analysisStatus === 'idle' || analysisStatus === 'error') && !modelPrediction.label && (
                                                            <div className="text-center py-4">
                                                                <p className="text-muted">{modelPrediction.explanation || 'Select an image and model.'}</p>
                                                            </div>
                                                        )}
                                                    </>
                                                )}
                                            </div>
                                            {(showAdvancedView && (
                                                <div className="p-2 mt-auto">
                                                    <Button variant='outline-success' onClick={setupGame}><h5>Let's play
                                                        a
                                                        minigame!</h5></Button>
                                                </div>
                                            ))}
                                            {/* --- Heatmap Controls Section --- */}
                                            <div className="border-top pt-3">
                                                <h5>Heatmap Controls</h5>
                                                {/* Show/Hide Toggle */}
                                                <div
                                                    className="text-center mb-3"> {/* Added text-center, moved mb-3 here */}
                                                    <Form.Check
                                                        type="switch"
                                                        id="heatmap-toggle"
                                                        label="Show Heatmap Overlay"
                                                        checked={showHeatmap}
                                                        onChange={(e) => setShowHeatmap(e.target.checked)}
                                                        disabled={!heatmapRawUrl || analysisStatus === 'loading'}
                                                        className="d-inline-block"
                                                    />
                                                </div>
                                                {/* Opacity Slider */}
                                                <Form.Group as={Row}
                                                            className="mb-1 align-items-center"> {/* Use Row, align vertically */}
                                                    <Form.Label column xs="6" sm="5"
                                                                className="text-end pe-2 mb-0"> {/* Column for label, right-aligned text */}
                                                        Overlay Transparency:
                                                    </Form.Label>
                                                    <Col xs="6" sm="7"> {/* Column for slider */}
                                                        {heatmapOpacity.toFixed(1)}
                                                        <Form.Range
                                                            min={0} max={1} step={0.1}
                                                            value={heatmapOpacity}
                                                            onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                                                            disabled={!heatmapRawUrl || analysisStatus === 'loading'}
                                                            title={`Overlay Transparency: ${heatmapOpacity.toFixed(1)}`} // Add tooltip
                                                        />
                                                    </Col>
                                                </Form.Group>
                                                {/* Threshold Slider */}
                                                {showAdvancedView && (
                                                    <Form.Group as={Row}
                                                                className="mb-1 align-items-center"> {/* Use Row, less margin */}
                                                        <Form.Label column xs="6" md="5" className="mb-0 text-end pe-2">
                                                            Focus Intensity Threshold:
                                                        </Form.Label>
                                                        <Col xs="6" md="7">
                                                            {threshold.toFixed(1)}
                                                            <Form.Range
                                                                min={0} max={1} step={0.1}
                                                                value={threshold}
                                                                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                                                disabled={!showHeatmap || !heatmapRawUrl || analysisStatus === 'loading'}
                                                            />
                                                        </Col>
                                                        {/* Description below the row */}
                                                        <small className="text-muted d-block mt-1">
                                                            Hide overlay areas the model considered less important
                                                            than this
                                                            level.
                                                        </small>
                                                    </Form.Group>
                                                )}
                                            </div>
                                            {!showAdvancedView && (
                                                <>
                                                    <div className="mt-3 border-top pt-3">
                                                        <Form.Label>Select Example Image</Form.Label>
                                                        <div
                                                            className="d-flex justify-content-center overflow-auto pb-2">
                                                            {EXAMPLE_IMAGES_BASE.map(example => (
                                                                <div key={example.key}
                                                                     className="text-center mx-1 flex-shrink-0">
                                                                    <img
                                                                        src={example.src}
                                                                        alt={example.label}
                                                                        className="rounded"
                                                                        style={{
                                                                            width: '80px',
                                                                            height: '80px',
                                                                            cursor: 'pointer',
                                                                            objectFit: 'cover',
                                                                            border: selectedImageType === example.key ? '3px solid #0d6efd' : '1px solid #dee2e6'
                                                                        }}
                                                                        onClick={() => selectExampleImage(example.key, example.src)}
                                                                        loading="lazy"
                                                                    />
                                                                    <div className="mt-1 text-primary"
                                                                         style={{fontSize: '0.8rem'}}>{example.label}</div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </>
                                            )}
                                        </Stack>
                                    </Col>
                                </Row>
                                {/* --- End of Side-by-Side Row --- */}


                                {/* --- Model Selection --- */}
                                <div className="border-top pt-3">
                                    <h5 className="mb-3">Select Model</h5>
                                    <Row className="g-2 mb-2 row-cols-2 row-cols-lg-4">
                                        {Object.values(MODELS_AVAILABLE).map((model) => (
                                            <Col key={model.key}>
                                                <Card
                                                    className={`h-100 p-2 model-select-card d-flex flex-column align-items-center ${selectedModelKey === model.key ? 'selected-model-card' : ''}`}
                                                    style={{cursor: 'pointer', transition: 'transform 0.2s'}}
                                                    onClick={() => handleModelSelect(model.key)}
                                                >
                                                    <div className="py-2 text-primary"><IconComponent
                                                        icon={model.icon}/></div>
                                                    <h6 className="mb-0">{model.name}</h6>
                                                    {showAdvancedView ? (
                                                        <>
                                                            <small
                                                                className="text-muted d-block">{model.description}</small>
                                                            <small className="text-muted" style={{
                                                                fontSize: '0.75rem',
                                                                marginTop: '4px'
                                                            }}>{model.laymanDescription}</small>
                                                        </>
                                                    ) : (<></>)}
                                                </Card>
                                            </Col>
                                        ))}
                                    </Row>
                                </div>


                            </Card.Body>
                        </Card>
                    </Col>

                    {/* ==================================== */}
                    {/* == Sidebar Column (lg={2}) == */}
                    {/* ==================================== */}
                    <Col lg={3}>
                        {/* --- Stack directly inside the column --- */}
                        <Stack gap={3}> {/* Increased gap slightly for stand-alone items */}
                            {showAdvancedView ? (
                                <>
                                    {/* --- Explanation Item 1 --- */}
                                    <div>
                                        {/* Clickable Header Card */}
                                        <Card
                                            className="p-2 model-select-card d-flex flex-row align-items-center shadow-sm" // Added shadow-sm for definition
                                            onClick={() => toggleExplanationSection('what')}
                                            aria-controls="explanation-what-collapse"
                                            aria-expanded={openExplanation === 'what'}
                                        >
                                            <h5 className="mb-0 flex-grow-1 text-center mx-2">What is Grad-CAM?</h5>
                                            <span style={{fontSize: '1.2em', fontWeight: 'bold'}}>
                                        {openExplanation === 'what' ? '−' : '+'}
                                    </span>
                                        </Card>
                                        {/* Collapsible Content */}
                                        <Collapse in={openExplanation === 'what'}>
                                            <div id="explanation-what-collapse">
                                                {/* Added border-start/end/bottom to connect visually */}
                                                <div
                                                    className="explanation-description-content border-start border-end border-bottom rounded-bottom">
                                                    <p>
                                                        Gradient-weighted Class Activation Mapping (Grad-CAM) helps
                                                        visualize which parts of an image an AI model focused on to
                                                        arrive at its conclusion, making the decision process less of a
                                                        'black box'. Essentially, it's like the AI is 'showing its work'
                                                        using a visual 'highlighter'. The model 'highlights' (often in
                                                        red/yellow) the image features that were most important for
                                                        classifying the object.
                                                    </p>
                                                    <p>
                                                        Being able to see this 'highlighting' is crucial. If the
                                                        highlighted areas align with logical features (like the model
                                                        focusing on a cat's face to identify it as a 'cat'), it
                                                        increases our trust in the AI's reasoning. Conversely, if the
                                                        heatmap highlights irrelevant parts (like the rug next to the
                                                        cat), it signals the AI might be using faulty logic or
                                                        shortcuts. Understanding the heatmap helps us verify the AI's
                                                        process and build more reliable models, much like checking
                                                        someone's math ensures their answer is sound.
                                                    </p>
                                                </div>
                                            </div>
                                        </Collapse>
                                    </div>
                                    {/* --- Explanation Item 2 --- */}
                                    <div>
                                        <Card
                                            className="p-2 model-select-card d-flex flex-row align-items-center shadow-sm"
                                            onClick={() => toggleExplanationSection('how')}
                                            aria-controls="explanation-how-collapse"
                                            aria-expanded={openExplanation === 'how'}
                                        >
                                            <h5 className="mb-0 flex-grow-1 text-center mx-2">Understanding the
                                                Heatmap</h5>
                                            <span style={{fontSize: '1.2em', fontWeight: 'bold'}}>
                                        {openExplanation === 'how' ? '−' : '+'}
                                    </span>
                                        </Card>
                                        <Collapse in={openExplanation === 'how'}>
                                            <div id="explanation-how-collapse">
                                                <div
                                                    className="explanation-description-content border-start border-end border-bottom rounded-bottom">
                                                    <p className="mb-2">The heatmap overlay visualizes these important
                                                        regions.
                                                        The 'jet' colormap is
                                                        used here:</p>
                                                    <div className="mb-1"><Badge bg="danger" className="me-1"
                                                                                 style={{width: '20px'}}> </Badge> Key
                                                        areas
                                                        influencing
                                                        the decision
                                                    </div>
                                                    <div className="mb-1"><Badge bg="warning" text="dark"
                                                                                 className="me-1"
                                                                                 style={{width: '20px'}}> </Badge> Moderately
                                                        important
                                                        areas
                                                    </div>
                                                    <div className="mb-1"><Badge bg="info" className="me-1"
                                                                                 style={{width: '20px'}}> </Badge> Less
                                                        important areas
                                                    </div>
                                                    <div className="mb-1"><Badge bg="primary" className="me-1"
                                                                                 style={{width: '20px'}}> </Badge> Areas
                                                        model
                                                        paid
                                                        least attention to
                                                    </div>
                                                </div>
                                            </div>
                                        </Collapse>
                                    </div>
                                </>
                            ) : (<></>)}


                            {/* --- Explanation Item 3 --- */}
                            <div>
                                <Card
                                    className="p-2 model-select-card d-flex flex-row align-items-center shadow-sm"
                                    onClick={() => toggleExplanationSection('using')}
                                    aria-controls="explanation-using-collapse"
                                    aria-expanded={openExplanation === 'using'}
                                >
                                    <h5 className="mb-0 flex-grow-1 text-center mx-2">Using This Tool</h5>
                                    <span style={{fontSize: '1.2em', fontWeight: 'bold'}}>
                                        {openExplanation === 'using' ? '−' : '+'}
                                    </span>
                                </Card>
                                <Collapse in={openExplanation === 'using'}>
                                    <div id="explanation-using-collapse">
                                        <div
                                            className="explanation-description-content border-start border-end border-bottom rounded-bottom">
                                            <ol className="mb-0 ps-3">
                                                <li>Pick an <strong class={'text-muted'}>Example Image</strong> below the main view.</li>
                                                <li>Choose a <strong class={'text-muted'}>Model</strong> using the cards.</li>
                                                <li>See the <strong class={'text-muted'}>AI's Prediction</strong> and read its <strong
                                                    class={'text-muted'}>Reasoning</strong>.</li>
                                                <li>Compare the reasoning to the <strong class={'text-muted'}>Heatmap Overlay</strong>.</li>
                                                <li>Adjust <strong class={'text-muted'}>Heatmap Controls</strong> to
                                                    explore further.
                                                </li>
                                            </ol>
                                        </div>
                                    </div>
                                </Collapse>
                            </div>

                        </Stack>
                    </Col>
                </Row>

                {/* Footer */}
                <footer className="mt-5 py-3 text-center border-top text-muted">
                    <p>Grad-CAM Visualization Tool © 2024</p>
                </footer>
            </Container>
            {/* == Multi-Stage Game Modal == */}
            {/* ================================ */}
            <Modal show={showGameModal} onHide={handleCloseGameModal} size="lg" centered backdrop="static">
                <Modal.Header closeButton>
                    <Modal.Title>
                        {gameStage === 1 ? "Stage 1: Explain the Focus" : "Stage 2: Spot Suspicious Focus"}
                    </Modal.Title>
                </Modal.Header>

                {/* --- Stage 1: Explain the Focus --- */}
                {gameStage === 1 && gameExamplesStage1.length > 0 && currentGameIndex < gameExamplesStage1.length ? (
                    (() => { // Immediately invoked function expression to use const
                        const currentExample = gameExamplesStage1[currentGameIndex];
                        // --- Look up options for the current example using the global lookup ---
                        const currentOptions = stage1OptionsLookup[currentExample.id] || stage1OptionsLookup['default'];

                        // --- GENERATE DYNAMIC INTRO TEXT
                        const introText = getStage1IntroText(currentExample, currentGameIndex);

                        return (
                            <>
                                <Modal.Body style={{position: 'relative', minHeight: '300px'}}>
                                    <div
                                        className="mb-3 mx-auto"
                                        style={{
                                            display: 'block',
                                            maxWidth: '600px',
                                        }}
                                    >
                                        <p className="text-center">Example {currentGameIndex + 1} of {gameExamplesStage1.length} (Stage
                                            1)</p>
                                        <p className="text-center">Prediction: <strong
                                            className="text-primary">{currentExample.topPredictions[0].className.replace(/_/g, ' ')}</strong>
                                        </p>
                                        {/* --- RENDER DYNAMIC INTRO TEXT --- */}
                                        <p
                                            className="text-center mb-3"
                                            dangerouslySetInnerHTML={{__html: introText}}
                                        />
                                    </div>
                                    {/* Image Container */}
                                    <div
                                        className="mb-3 mx-auto"
                                        style={{
                                            display: 'block',
                                            maxWidth: '350px',
                                        }}
                                    >
                                        { /* Conditional img tag inside */}
                                        {currentExample.heatmapOverlayUrl ? (
                                            <img
                                                src={currentExample.heatmapOverlayUrl}
                                                alt={`Overlay for ${currentExample.id}`}
                                                className="rounded border"
                                                style={{
                                                    display: 'block', // Prevent extra space below image
                                                    width: '100%',    // Fill the container's width (up to maxWidth)
                                                    height: 'auto'     // Maintain aspect ratio
                                                }}
                                            />
                                        ) : (
                                            <img
                                                src={currentExample.originalImageUrl}
                                                alt={`Original for ${currentExample.id}`}
                                                className="rounded border"
                                                style={{
                                                    display: 'block',
                                                    width: '100%',
                                                    height: 'auto'
                                                }}
                                            />
                                        )}
                                    </div>

                                    {/* Answer Options (Buttons) */}
                                    {!showGameAnswer && (
                                        <Stack gap={2} className="align-items-center mt-3">
                                            {currentOptions.map(option => (
                                                <Button key={option.key} variant="outline-primary" size="lg"
                                                        onClick={() => handleGameStage1Answer(option.key)}> {/* Pass option.key */}
                                                    {option.key}: {option.text}
                                                </Button>
                                            ))}
                                        </Stack>
                                    )}

                                    {/* Feedback Area */}
                                    {showGameAnswer && (
                                        <Alert variant={gameFeedback.startsWith('Correct') ? 'success' : 'danger'}
                                               className="text-center mt-3">{gameFeedback}</Alert>)}
                                </Modal.Body>
                                <Modal.Footer>
                                    {showGameAnswer && (
                                        <Button variant="primary" onClick={handleNextGameStep}>Next</Button>)}
                                    <Button variant="secondary" onClick={handleCloseGameModal}>Close Game</Button>
                                </Modal.Footer>
                            </>
                        );
                    })() // End of IIFE
                ) : null}

                {/* --- Stage 2: Anomaly Detector --- */}
                {gameStage === 2 && gameExamplesStage2.length > 0 && currentGameIndex < gameExamplesStage2.length ? (
                    (() => { // IIFE
                        const currentExample = gameExamplesStage2[currentGameIndex];
                        return (
                            <>
                                <Modal.Body>
                                    {/* --- Generate and Render Dynamic Intro Text --- */}
                                    {(() => { // Use IIFE to get variables in scope
                                        const currentExample = gameExamplesStage2[currentGameIndex];
                                        const imageKey = currentExample.id.split('_').slice(1).join('_');
                                        const trueLabel = TRUE_IMAGE_LABELS[imageKey] || imageKey;
                                        const introText = getStage2IntroText(currentExample, currentGameIndex);
                                        return (
                                            <>
                                                <p className="text-center">Example {currentGameIndex + 1} of {gameExamplesStage2.length} (Stage
                                                    2)</p>
                                                {/* Prediction and True Label (as before) */}
                                                <p className="text-center">The AI predicted: <strong
                                                    className="text-primary">{currentExample.topPredictions[0].className.replace(/_/g, ' ')}</strong>
                                                </p>
                                                <p className="text-center text-muted"
                                                   style={{fontSize: '0.9rem', marginTop: '-0.75rem'}}>
                                                    (Actual Subject: <span className="fw-bold">{trueLabel}</span>)
                                                </p>
                                                {/* Dynamic Instruction Text */}
                                                <p
                                                    className="text-center text-muted mb-3"
                                                    dangerouslySetInnerHTML={{__html: introText}} // Use dangerouslySetInnerHTML
                                                />
                                            </>
                                        );
                                    })()}
                                    {/* --- End Intro Text Generation/Rendering --- */}
                                    {/* Image Container */}
                                    <div
                                        className="mb-3 mx-auto"
                                        style={{
                                            display: 'block',
                                            maxWidth: '350px',
                                        }}
                                    >
                                        { /* Conditional img tag inside */}
                                        {currentExample.heatmapOverlayUrl ? (
                                            <img
                                                src={currentExample.heatmapOverlayUrl}
                                                alt={`Overlay for ${currentExample.id}`}
                                                className="rounded border"
                                                style={{
                                                    display: 'block', // Prevent extra space below image
                                                    width: '100%',    // Fill the container's width (up to maxWidth)
                                                    height: 'auto'     // Maintain aspect ratio
                                                }}
                                            />
                                        ) : (
                                            <img
                                                src={currentExample.originalImageUrl}
                                                alt={`Original for ${currentExample.id}`}
                                                className="rounded border"
                                                style={{
                                                    display: 'block',
                                                    width: '100%',
                                                    height: 'auto'
                                                }}
                                            />
                                        )}
                                    </div>

                                    {/* Judgment Buttons */}
                                    {!showGameAnswer && (
                                        <div className="d-flex justify-content-center gap-3 mt-3">
                                            <Button variant="outline-success" size="lg"
                                                    onClick={() => handleGameStage2Answer('logical')}>Logical
                                                👍</Button>
                                            <Button variant="outline-warning" size="lg"
                                                    onClick={() => handleGameStage2Answer('suspicious')}>Suspicious
                                                🤔</Button>
                                        </div>
                                    )}

                                    {/* Feedback Area */}
                                    {showGameAnswer && (
                                        <Alert variant={gameFeedback.startsWith('Correct') ? 'success' : 'danger'}
                                               className="text-center mt-3">{gameFeedback}</Alert>)}
                                </Modal.Body>
                                <Modal.Footer>
                                    {showGameAnswer && (
                                        <Button variant="primary" onClick={handleNextGameStep}>
                                            {currentGameIndex < gameExamplesStage2.length - 1 ? "Next Example" : "Finish Game"}
                                        </Button>
                                    )}
                                    <Button variant="secondary" onClick={handleCloseGameModal}>Close Game</Button>
                                </Modal.Footer>
                            </>
                        );
                    })() // End of IIFE
                ) : null}

                {/* Fallback / Loading state for modal if needed */}
                {!(gameStage === 1 && gameExamplesStage1.length > 0 && currentGameIndex < gameExamplesStage1.length) && !(gameStage === 2 && gameExamplesStage2.length > 0 && currentGameIndex < gameExamplesStage2.length) && showGameModal && (
                    <Modal.Body><p className="text-center">Loading game examples...</p></Modal.Body>
                )}

            </Modal>
            {/* --- Tooltip Div (Rendered outside main flow, positioned absolutely) --- */}
            {showAdvancedView && hoverInfo && showHeatmap && hoverInfo.intensity >= threshold && (
                <div
                    style={{
                        position: 'absolute',
                        left: `${hoverInfo.pageX + 15}px`, // Offset slightly from cursor
                        top: `${hoverInfo.pageY + 15}px`,
                        padding: '5px 8px',
                        background: 'rgba(0, 0, 0, 0.75)',
                        color: 'white',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        zIndex: 1050, // Ensure it's above other elements
                        pointerEvents: 'none', // Allow clicks to pass through
                        whiteSpace: 'nowrap',
                    }}
                >
                    Intensity: {hoverInfo.intensity.toFixed(2)}
                </div>
            )}
        </div>
    );
}