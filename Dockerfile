# ==============================
# 1️⃣ Base image
# ==============================
FROM node:20-slim

# Install required system packages for ONNX Runtime and image processing
RUN apt-get update && apt-get install -y \
    python3 \
    build-essential \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 2️⃣ Set working directory
# ==============================
WORKDIR /app

# ==============================
# 3️⃣ Copy package files
# ==============================
COPY package*.json ./

# Install production dependencies only
RUN npm install --omit=dev

# ==============================
# 4️⃣ Copy app files
# ==============================
COPY . .

# ==============================
# 5️⃣ Environment setup
# ==============================
ENV NODE_ENV=production
ENV PORT=3000

# Make sure model path exists
RUN mkdir -p models uploads

# ==============================
# 6️⃣ Expose port and start
# ==============================
EXPOSE 3000
CMD ["npm", "start"]
