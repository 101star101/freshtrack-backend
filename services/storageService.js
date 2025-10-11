const fs = require('fs');
const path = require('path');
require('dotenv').config();

// ==============================
// Default storage data
// ==============================
const DEFAULT_STORAGE_DATA = {
  // Fresh Fruits
  "Fresh_Apple": {
    "storage": "Refrigerate in crisper drawer",
    "shelf_life": 14,
    "tips": "Store away from other fruits to prevent ripening",
    "signs_of_spoilage": "Soft spots, mold, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Banana": {
    "storage": "Store at room temperature until ripe, then refrigerate",
    "shelf_life": 7,
    "tips": "Keep away from other fruits, wrap stem in plastic",
    "signs_of_spoilage": "Black spots, mushy texture, strong odor",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Orange": {
    "storage": "Store at room temperature or refrigerate",
    "shelf_life": 14,
    "tips": "Store in mesh bag for air circulation",
    "signs_of_spoilage": "Soft spots, mold, dry texture",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Manggo": {
    "storage": "Store at room temperature until ripe, then refrigerate",
    "shelf_life": 7,
    "tips": "Store away from other fruits",
    "signs_of_spoilage": "Soft spots, mold, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Strawberry": {
    "storage": "Refrigerate in original container",
    "shelf_life": 7,
    "tips": "Don't wash until ready to use",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },

  // Fresh Vegetables
  "Fresh_Potato": {
    "storage": "Store in cool, dark, dry place",
    "shelf_life": 30,
    "tips": "Keep away from onions",
    "signs_of_spoilage": "Green spots, soft spots, sprouting",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Carrot": {
    "storage": "Refrigerate in plastic bag",
    "shelf_life": 21,
    "tips": "Remove green tops before storing",
    "signs_of_spoilage": "Soft texture, white spots, mold",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Pepper": {
    "storage": "Refrigerate in plastic bag",
    "shelf_life": 7,
    "tips": "Store in crisper drawer",
    "signs_of_spoilage": "Soft spots, mold, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Cucumber": {
    "storage": "Refrigerate in plastic bag",
    "shelf_life": 7,
    "tips": "Store away from ethylene-producing fruits",
    "signs_of_spoilage": "Soft spots, mold, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Okra": {
    "storage": "Refrigerate in plastic bag",
    "shelf_life": 7,
    "tips": "Store in high humidity drawer",
    "signs_of_spoilage": "Soft texture, mold, wrinkled skin",
    "status": "Fresh",
    "waste_disposal": null
  },

  // Fresh Meats
  "Fresh_Beef": {
    "storage": "Refrigerate at 32-40°F (0-4°C)",
    "shelf_life": 3,
    "tips": "Store on bottom shelf to prevent cross-contamination",
    "signs_of_spoilage": "Gray color, slimy texture, strong odor",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Chicken": {
    "storage": "Refrigerate at 32-40°F (0-4°C)",
    "shelf_life": 2,
    "tips": "Store on bottom shelf to prevent cross-contamination",
    "signs_of_spoilage": "Gray color, slimy texture, strong odor",
    "status": "Fresh",
    "waste_disposal": null
  },
  "Fresh_Pork": {
    "storage": "Refrigerate at 32-40°F (0-4°C)",
    "shelf_life": 3,
    "tips": "Store on bottom shelf to prevent cross-contamination",
    "signs_of_spoilage": "Gray color, slimy texture, strong odor",
    "status": "Fresh",
    "waste_disposal": null
  },

  // Rotten Items
  "Rotten_Apple": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Banana": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Black spots, mushy texture, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Orange": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, dry texture, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Manggo": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Strawberry": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Beef": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Gray color, slimy texture, strong foul odor",
    "status": "Rotten",
    "waste_disposal": "Double-bag in plastic and dispose in trash. Do not compost meat products."
  },
  "Rotten_Chicken": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Gray color, slimy texture, strong foul odor",
    "status": "Rotten",
    "waste_disposal": "Double-bag in plastic and dispose in trash. Do not compost meat products."
  },
  "Rotten_Pork": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Gray color, slimy texture, strong foul odor",
    "status": "Rotten",
    "waste_disposal": "Double-bag in plastic and dispose in trash. Do not compost meat products."
  }
};

// ==============================
// StorageService Class
// ==============================
class StorageService {
  constructor() {
    const defaultPath = path.join(__dirname, '../data/storage_data.json');
    this.storageDataPath = process.env.STORAGE_PATH || defaultPath;
    this.storageData = this.loadStorageData();
  }

  loadStorageData() {
    try {
      if (fs.existsSync(this.storageDataPath)) {
        const data = fs.readFileSync(this.storageDataPath, 'utf8');
        return JSON.parse(data);
      } else {
        console.warn(`⚠️ Storage file not found. Creating new file at ${this.storageDataPath}`);
        const dir = path.dirname(this.storageDataPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(this.storageDataPath, JSON.stringify(DEFAULT_STORAGE_DATA, null, 2));
        return DEFAULT_STORAGE_DATA;
      }
    } catch (err) {
      console.error('❌ Error loading storage data:', err);
      return DEFAULT_STORAGE_DATA;
    }
  }

  normalizeLabel(label) {
    if (!label) return '';
    let formatted = label.trim();

    // Normalize prefixes
    if (formatted.toLowerCase().startsWith('fresh_'))
      formatted = 'Fresh_' + formatted.slice(6);
    else if (formatted.toLowerCase().startsWith('rotten_'))
      formatted = 'Rotten_' + formatted.slice(7);

    // Capitalize second word
    const parts = formatted.split('_');
    if (parts.length > 1) {
      parts[1] = parts[1].charAt(0).toUpperCase() + parts[1].slice(1).toLowerCase();
      formatted = parts.join('_');
    }

    return formatted;
  }

  getStorageData(itemName) {
    if (!itemName || typeof itemName !== 'string') {
      console.warn('⚠️ Invalid item name:', itemName);
      return null;
    }

    const normalized = this.normalizeLabel(itemName);

    if (this.storageData[normalized]) return this.storageData[normalized];

    for (const [key, value] of Object.entries(this.storageData)) {
      if (key.toLowerCase() === normalized.toLowerCase()) return value;
    }

    console.warn(`⚠️ No storage data found for "${itemName}"`);
    return null;
  }

  getAllStorageData() {
    return this.storageData;
  }
}

const storageService = new StorageService();

module.exports = {
  StorageService,
  getStorageData: (itemName) => storageService.getStorageData(itemName),
  storageService
};