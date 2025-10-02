# Forex Trading Signal Frontend

A modern React application for displaying forex trading signals with professional candlestick charts and AI predictions.

## ✅ **Frontend Setup Checklist**

### 🔧 **Development Environment**
- [x] Node.js 16+ installed
- [x] npm or yarn package manager
- [x] Git repository access
- [x] Code editor (VS Code recommended)

### 📦 **Dependencies Installation**
- [x] React 18+ installed
- [x] Recharts library for charting
- [x] Axios for API calls
- [x] React Router for navigation
- [x] CSS modules or styled-components

### 🎨 **Core Components**
- [x] App.js - Main application component
- [x] CandlestickChart.js - Custom candlestick chart with predictions
- [x] Signal cards displaying probability and direction
- [x] Chart type selector (Custom/TradingView)
- [x] Backtesting interface
- [x] Responsive layout components

### 📊 **Chart Features**
- [x] Candlestick chart display with OHLC data
- [x] Gold prediction candles for AI signals
- [x] Star indicators for prediction points
- [x] Professional spacing and layout
- [x] Enhanced tooltips with detailed information
- [x] Current data display (2025 dates)
- [x] Zoom and pan functionality

### 🔗 **API Integration**
- [x] Axios configured for backend communication
- [x] Signal data fetching from Django API
- [x] Historical data loading
- [x] Backtesting results display
- [x] Error handling for API failures
- [x] Loading states and user feedback
- [x] Holloway algorithm parity fields (weighted counts, critical level flags) surfaced in UI payloads

### 🎯 **TradingView Integration Options**
- [x] react-tradingview-widget setup (optional)
- [x] lightweight-charts integration (optional)
- [x] react-financial-charts support (optional)
- [x] Chart type switching functionality

### 📱 **UI/UX Features**
- [x] Modern, professional design
- [x] Mobile-responsive layout
- [x] Dark/light theme support (planned)
- [x] Intuitive navigation
- [x] Real-time data updates
- [x] Loading indicators and error messages

### 🧪 **Testing & Validation**
- [x] Component rendering tests
- [x] API integration tests
- [x] Chart display validation
- [x] Mobile responsiveness testing
- [x] Cross-browser compatibility

## 🚀 **Quick Start**

### Prerequisites
- Node.js 16 or higher
- npm or yarn
- Backend API running (Django on port 8000)

### Installation

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start development server**
```bash
npm start
```

The application will open at [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## 📊 **Available Scripts**

### `npm start`
Runs the app in development mode with hot reloading.

### `npm test`
Launches the test runner in interactive watch mode.

### `npm run build`
Builds the app for production to the `build` folder.

### `npm run eject`
**Note: This is a one-way operation!**

Removes the single build dependency and copies all configuration files into the project.

## 🏗️ **Project Structure**

```
frontend/
├── public/
│   ├── index.html
│   ├── favicon.ico
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── CandlestickChart.js
│   │   ├── SignalCard.js
│   │   └── BacktestInterface.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── setupTests.js
├── package.json
└── README.md
```

## 🔧 **Configuration**

### API Configuration
Update the API base URL in components that make API calls:

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';
```

### Environment Variables
Create a `.env` file for environment-specific configuration:

```env
REACT_APP_API_URL=http://localhost:8000/api
REACT_APP_CHART_TYPE=default
```

## 🎨 **Chart Customization**

### Custom Chart Features
- Gold outlined prediction candles
- Star markers for AI predictions
- Enhanced tooltips with OHLC data
- Professional color scheme
- Responsive design

### TradingView Integration
For production-grade charts, consider integrating:

1. **react-tradingview-widget**: Full TradingView experience
2. **lightweight-charts**: High-performance alternative
3. **react-financial-charts**: Advanced technical analysis

## 📱 **Mobile Optimization**

- Responsive grid layout
- Touch-friendly controls
- Optimized chart interactions
- Mobile-specific navigation

## 🧪 **Testing**

### Running Tests
```bash
npm test
```

### Test Coverage
- Component rendering
- API integration
- User interactions
- Error handling

## 🚀 **Deployment**

### Build Process
```bash
npm run build
```

### Deployment Options
- **Netlify**: Drag and drop the `build` folder
- **Vercel**: Connect GitHub repository
- **Namecheap**: Upload build files via FTP
- **AWS S3 + CloudFront**: Static hosting

### Environment Configuration
Set production environment variables:
- API endpoints
- Analytics tracking
- Error reporting

## 🐛 **Troubleshooting**

### Common Issues

1. **API Connection Failed**
   - Ensure Django backend is running on port 8000
   - Check CORS configuration
   - Verify API endpoints are accessible

2. **Charts Not Displaying**
   - Check browser console for errors
   - Verify data format from API
   - Ensure Recharts dependencies are installed

3. **Build Failures**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify all dependencies are listed in package.json

## 🤝 **Contributing**

1. Follow React best practices
2. Use functional components with hooks
3. Implement proper error boundaries
4. Add tests for new components
5. Follow existing code style

## 📄 **License**

This project is part of the Forex Trading System and follows the same license terms.

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
