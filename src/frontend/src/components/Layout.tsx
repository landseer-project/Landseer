import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  TooltipProvider,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from '@/components/ui/tooltip';
import {
  LayoutDashboard,
  ListTodo,
  GitBranch,
  Users,
  Wrench,
  Activity,
  Settings,
  Moon,
  Sun,
  Menu,
  Database,
  BarChart3,
} from 'lucide-react';
import { useState, useEffect } from 'react';

interface NavItem {
  title: string;
  href: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    href: '/',
    icon: <LayoutDashboard className="h-5 w-5" />,
  },
  {
    title: 'Tasks',
    href: '/tasks',
    icon: <ListTodo className="h-5 w-5" />,
  },
  {
    title: 'Workflows',
    href: '/workflows',
    icon: <GitBranch className="h-5 w-5" />,
  },
  {
    title: 'Workers',
    href: '/workers',
    icon: <Users className="h-5 w-5" />,
  },
  {
    title: 'Tools',
    href: '/tools',
    icon: <Wrench className="h-5 w-5" />,
  },
  {
    title: 'Registry',
    href: '/registry',
    icon: <Database className="h-5 w-5" />,
  },
  {
    title: 'Metrics',
    href: '/metrics',
    icon: <BarChart3 className="h-5 w-5" />,
  },
];

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const [isDark, setIsDark] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    // Check system preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    const isDarkMode = savedTheme === 'dark' || (!savedTheme && prefersDark);
    setIsDark(isDarkMode);
    document.documentElement.classList.toggle('dark', isDarkMode);
  }, []);

  const toggleTheme = () => {
    const newDark = !isDark;
    setIsDark(newDark);
    document.documentElement.classList.toggle('dark', newDark);
    localStorage.setItem('theme', newDark ? 'dark' : 'light');
  };

  return (
    <TooltipProvider>
      <div className="flex min-h-screen bg-background">
        {/* Sidebar */}
        <aside
          className={cn(
            'fixed left-0 top-0 z-40 h-screen border-r bg-card transition-all duration-300',
            sidebarOpen ? 'w-64' : 'w-16'
          )}
        >
          <div className="flex h-full flex-col">
            {/* Logo */}
            <div className="flex h-16 items-center justify-between border-b px-4">
              {sidebarOpen && (
                <Link to="/" className="flex items-center gap-2">
                  <Activity className="h-6 w-6 text-primary" />
                  <span className="text-xl font-bold">Landseer</span>
                </Link>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className={cn(!sidebarOpen && 'mx-auto')}
              >
                <Menu className="h-5 w-5" />
              </Button>
            </div>

            {/* Navigation */}
            <ScrollArea className="flex-1 py-4">
              <nav className="space-y-1 px-2">
                {navItems.map((item) => {
                  const isActive = location.pathname === item.href;
                  return sidebarOpen ? (
                    <Link
                      key={item.href}
                      to={item.href}
                      className={cn(
                        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      )}
                    >
                      {item.icon}
                      {item.title}
                    </Link>
                  ) : (
                    <Tooltip key={item.href}>
                      <TooltipTrigger asChild>
                        <Link
                          to={item.href}
                          className={cn(
                            'flex items-center justify-center rounded-lg p-3 transition-colors',
                            isActive
                              ? 'bg-primary text-primary-foreground'
                              : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                          )}
                        >
                          {item.icon}
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent side="right">{item.title}</TooltipContent>
                    </Tooltip>
                  );
                })}
              </nav>
            </ScrollArea>

            {/* Footer */}
            <div className="border-t p-4">
              <div className={cn('flex', sidebarOpen ? 'justify-between' : 'justify-center')}>
                {sidebarOpen && (
                  <Button variant="ghost" size="icon">
                    <Settings className="h-5 w-5" />
                  </Button>
                )}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon" onClick={toggleTheme}>
                      {isDark ? (
                        <Sun className="h-5 w-5" />
                      ) : (
                        <Moon className="h-5 w-5" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side={sidebarOpen ? 'top' : 'right'}>
                    Toggle theme
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main
          className={cn(
            'flex-1 transition-all duration-300',
            sidebarOpen ? 'ml-64' : 'ml-16'
          )}
        >
          <div className="container mx-auto p-6">{children}</div>
        </main>
      </div>
    </TooltipProvider>
  );
}
