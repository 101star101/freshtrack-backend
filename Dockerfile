# ==============================
# 1️⃣ Base Image
# ==============================
FROM node:20-slim

# Install dependencies for image & ONNX processing
RUN apt-get update && apt-get install -y \
    python3 \
    build-essential \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 2️⃣ Working Directory
# ==============================
WORKDIR /app

# ==============================
# 3️⃣ Copy Package Files
# ==============================
COPY package*.json ./

# Install production dependencies
RUN npm install --omit=dev

# ==============================
# 4️⃣ Copy Application Files
# ==============================
COPY . .

# ✅ Ensure .env is included
COPY .env .env

# Create necessary folders
RUN mkdir -p models uploads data

# ==============================
# 5️⃣ Environment Setup
# ==============================
ENV NODE_ENV=production
ENV PORT=3000

# ==============================
# 6️⃣ Expose & Start
# ==============================
EXPOSE 3000
CMD ["npm", "start"]
