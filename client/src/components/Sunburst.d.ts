declare module '@/components/Sunburst' {
  import { FC } from 'react';
  
  interface SunburstProps {
    onSelect?: (name: string) => void;
  }
  
  const Sunburst: FC<SunburstProps>;
  export default Sunburst;
}

