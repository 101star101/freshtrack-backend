const fs = require('fs');
const path = require('path');

// Default storage data for fresh and rotten food items
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
  
  // Rotten Fruits
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
  
  // Rotten Vegetables
  "Rotten_Potato": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Green spots, soft spots, mold, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Carrot": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Soft texture, white spots, mold, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Pepper": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Cucumber": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Mold, soft spots, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  "Rotten_Okra": {
    "storage": "DISPOSE IMMEDIATELY",
    "shelf_life": 0,
    "tips": "Do not consume - dispose safely",
    "signs_of_spoilage": "Soft texture, mold, wrinkled skin, strong odor",
    "status": "Rotten",
    "waste_disposal": "Compost if no mold, otherwise dispose in sealed bag in trash"
  },
  
  // Rotten Meats
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

class StorageService {
  constructor() {
    this.storageDataPath = path.join(__dirname, '../data/storage_data.json');
    this.storageData = this.loadStorageData();
  }

  loadStorageData() {
    try {
      if (fs.existsSync(this.storageDataPath)) {
        const data = fs.readFileSync(this.storageDataPath, 'utf8');
        return JSON.parse(data);
      } else {
        // Create data directory if it doesn't exist
        const dataDir = path.dirname(this.storageDataPath);
        if (!fs.existsSync(dataDir)) {
          fs.mkdirSync(dataDir, { recursive: true });
        }
        
        // Write default data
        fs.writeFileSync(this.storageDataPath, JSON.stringify(DEFAULT_STORAGE_DATA, null, 2));
        return DEFAULT_STORAGE_DATA;
      }
    } catch (error) {
      console.error('Error loading storage data:', error);
      return DEFAULT_STORAGE_DATA;
    }
  }

  getStorageData(itemName) {
    const normalizedName = itemName.toLowerCase().trim();
    
    // Direct match
    if (this.storageData[normalizedName]) {
      return this.storageData[normalizedName];
    }
    
    // Partial match
    for (const [key, value] of Object.entries(this.storageData)) {
      if (key.includes(normalizedName) || normalizedName.includes(key)) {
        return value;
      }
    }
    
    // Return default for unknown items
    return {
      "storage": "Store in cool, dry place",
      "shelf_life": 7,
      "tips": "Check regularly for signs of spoilage",
      "signs_of_spoilage": "Mold, soft spots, off odor",
      "status": "Unknown",
      "waste_disposal": null
    };
  }

  calculateRemainingLife(itemName, detectionDate) {
    const storageInfo = this.getStorageData(itemName);
    const detectionTime = new Date(detectionDate);
    const currentTime = new Date();
    const timeElapsed = Math.floor((currentTime - detectionTime) / (1000 * 60 * 60 * 24)); // Days
    
    const remainingLife = Math.max(0, storageInfo.shelf_life - timeElapsed);
    
    return {
      ...storageInfo,
      remaining_life: remainingLife,
      detection_date: detectionDate,
      days_elapsed: timeElapsed
    };
  }

  updateStorageData(itemName, newData) {
    const normalizedName = itemName.toLowerCase().trim();
    this.storageData[normalizedName] = { ...this.storageData[normalizedName], ...newData };
    
    try {
      fs.writeFileSync(this.storageDataPath, JSON.stringify(this.storageData, null, 2));
      return true;
    } catch (error) {
      console.error('Error updating storage data:', error);
      return false;
    }
  }

  getAllStorageData() {
    return this.storageData;
  }

  searchItems(query) {
    const normalizedQuery = query.toLowerCase().trim();
    const results = [];
    
    for (const [key, value] of Object.entries(this.storageData)) {
      if (key.includes(normalizedQuery) || 
          value.storage.toLowerCase().includes(normalizedQuery) ||
          value.tips.toLowerCase().includes(normalizedQuery)) {
        results.push({ item: key, ...value });
      }
    }
    
    return results;
  }
}

const storageService = new StorageService();

function getStorageData(itemName) {
  return storageService.getStorageData(itemName);
}

module.exports = {
  StorageService,
  getStorageData,
  storageService
};
