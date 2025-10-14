# GPU Calculator - Design Style Guide

## Design Philosophy

### Visual Language
- **Technical Precision**: Clean, engineering-focused aesthetic that conveys accuracy and reliability
- **Professional Authority**: Sophisticated color palette and typography that builds trust with technical users
- **Data-Driven Clarity**: Clear visual hierarchy that makes complex calculations immediately understandable
- **Modern Efficiency**: Streamlined interface that respects users' time and expertise

### Color Palette
- **Primary**: Deep Navy (#1a2332) - Professional, trustworthy foundation
- **Secondary**: Electric Blue (#00d4ff) - Technical precision, data visualization
- **Accent**: Amber (#ffb347) - Warnings, optimization suggestions
- **Success**: Sage Green (#7fb069) - Optimal configurations, success states
- **Background**: Charcoal (#2d3748) - Modern, technical backdrop
- **Text Light**: Soft Gray (#e2e8f0) - High contrast readability
- **Text Dark**: Deep Charcoal (#1a202c) - For light backgrounds

### Typography
- **Display Font**: "JetBrains Mono" - Technical, monospace for data and calculations
- **Heading Font**: "Inter" - Modern, clean sans-serif for UI elements
- **Body Font**: "Inter" - Consistent, readable for descriptions and content
- **Code Font**: "JetBrains Mono" - Technical specifications and formulas

## Visual Effects & Animations

### Core Libraries Integration
1. **Anime.js**: Smooth transitions for calculation updates and UI state changes
2. **ECharts.js**: Professional data visualizations for memory breakdown and performance metrics
3. **Splitting.js**: Text animations for dynamic value updates
4. **Typed.js**: Typewriter effect for technical explanations
5. **p5.js**: Subtle background particle system representing data flow
6. **Pixi.js**: GPU-accelerated visual effects for performance indicators
7. **Matter.js**: Physics-based animations for optimization suggestions

### Background Effects
- **Particle System**: Subtle floating particles using p5.js, representing data streams and computational flow
- **Grid Pattern**: Technical grid background with animated connection lines
- **Depth Layers**: Parallax scrolling with multiple background layers for visual depth

### Interactive Elements
- **Slider Animations**: Smooth value transitions with visual feedback
- **Card Hover Effects**: 3D tilt and shadow expansion on hover
- **Button States**: Morphing colors and subtle glow effects
- **Progress Bars**: Animated fills with gradient transitions

### Text Effects
- **Value Counters**: Animated number counting for calculation results
- **Highlight Pulse**: Subtle pulsing highlights for important metrics
- **Typewriter Reveals**: Technical explanations appearing with typewriter effect
- **Split Letter Animations**: Staggered letter animations for headings

## Layout & Structure

### Grid System
- **Desktop**: 12-column grid with 24px gutters
- **Tablet**: 8-column grid with 20px gutters  
- **Mobile**: 4-column grid with 16px gutters

### Component Hierarchy
1. **Header**: Navigation and branding (60px height)
2. **Hero Section**: Calculator introduction (200px height)
3. **Main Calculator**: Interactive controls and inputs (flexible)
4. **Results Panel**: Real-time calculations and recommendations (flexible)
5. **Footer**: Technical information and links (80px height)

### Spacing System
- **Base Unit**: 8px
- **Component Spacing**: 16px, 24px, 32px, 48px
- **Section Spacing**: 64px, 96px, 128px
- **Page Margins**: 24px (mobile), 48px (desktop)

## Component Design

### Calculator Interface
- **Input Cards**: Rounded corners (8px), subtle shadows, hover elevation
- **Sliders**: Custom styled with gradient tracks and animated thumbs
- **Dropdowns**: Clean, minimal design with smooth open/close animations
- **Toggle Switches**: Smooth state transitions with color morphing

### Results Display
- **Memory Breakdown**: Donut chart with animated segments
- **Performance Metrics**: Bar charts with gradient fills
- **GPU Recommendations**: Card-based layout with comparison indicators
- **Optimization Tips**: Highlighted suggestion boxes with icons

### Data Visualization
- **Color Scheme**: Consistent with brand palette, maximum 4 colors per chart
- **Animations**: Smooth data transitions, staggered reveals
- **Interactivity**: Hover states, tooltips, and drill-down capabilities
- **Accessibility**: High contrast ratios and clear visual hierarchy

## Responsive Behavior

### Breakpoints
- **Mobile**: 320px - 768px
- **Tablet**: 768px - 1024px
- **Desktop**: 1024px - 1440px
- **Large Desktop**: 1440px+

### Adaptive Features
- **Mobile-First**: Touch-friendly controls with adequate spacing
- **Progressive Enhancement**: Advanced animations on larger screens
- **Flexible Layouts**: Grid systems that adapt to content requirements
- **Performance Optimization**: Reduced animations on lower-end devices

## Accessibility Considerations

### Color & Contrast
- **Minimum Ratio**: 4.5:1 for normal text, 3:1 for large text
- **Color Independence**: No information conveyed by color alone
- **Focus Indicators**: Clear, high-contrast focus states

### Interaction Design
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Proper ARIA labels and descriptions
- **Motion Preferences**: Respect user's reduced motion preferences
- **Touch Targets**: Minimum 44px touch targets on mobile

## Technical Implementation

### CSS Architecture
- **Custom Properties**: CSS variables for consistent theming
- **Component-Based**: Modular CSS with BEM methodology
- **Performance**: Optimized animations using transform and opacity
- **Fallbacks**: Graceful degradation for older browsers

### Animation Performance
- **GPU Acceleration**: Transform-based animations for smooth performance
- **Reduced Motion**: Respect user preferences for motion sensitivity
- **Progressive Enhancement**: Core functionality without JavaScript
- **Loading States**: Skeleton screens and progressive loading indicators