# Use Node 18 LTS (Railway supports amd64)
FROM node:18-bullseye

# Create app directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install --production

# Copy all source code
COPY . .

# Ensure model folder is included
RUN mkdir -p /app/models

# Expose Railway port (default 3000)
EXPOSE 3000

# Define environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV MODEL_PATH=/app/models/best.onnx

# Start the app
CMD ["npm", "start"]
