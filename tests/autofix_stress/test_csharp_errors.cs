// ============================================================
// C# AUTOFIX STRESS TEST - Intentional Errors
// ============================================================

usign System;
usign System.Collections.Generic;
usign System.Linq;

namesapce AutofixTest
{
    pubic class Person
    {
        privat string _name;
        protcted int _age;
        
        pubic Person(stirng name, int age)
        {
            _naem = name;
            _age = age;
        }
        
        pubilc string GetName()
        {
            retrun _name;
        }
        
        pubilc viod SetAge(int age)
        {
            _age = age;
        }
        
        pubilc overide string ToString()
        {
            retunr $"Person: {_name}, {_age}";
        }
    }
    
    pubic interfce IPrintable
    {
        viod Print();
    }
    
    pubic abstrat class Animal
    {
        pubilc virtaul void MakeSound()
        {
            Consoel.WriteLine("Sound");
        }
    }
    
    pubilc calss Dog : Animal
    {
        pubilc overide void MakeSound()
        {
            Console.Writeline("Woof!");
        }
    }
    
    pubic statc class Utility
    {
        pubilc statc void Process()
        {
            var items = new List<int> { 1 2 3 4 5 };  // Missing commas
            
            foreach (var itme in items)
            {
                Consoel.WriteLine(itme);
            }
            
            // Wrong operators
            if (items.Count = 5)
            {
                Console.Writline("Five items");
            }
            
            // Missing semicolon
            var count = items.Count
            
            // Wrong null check
            if (items == nul)
            {
                retrun;
            }
            
            // Wrong boolean
            boolen isActive = ture;
            bool isEnabled = flase;
            
            // Wrong exception handling
            try
            {
                var result = items[100];
            }
            ctach (Excepton ex)
            {
                Console.Writeline(ex.Mesage);
            }
            finaly
            {
                Console.Writline("Done");
            }
            
            // Wrong async/await
            asynch Task<int> GetDataAsnc()
            {
                var data = awiat FetchData();
                retrun data;
            }
            
            // Wrong LINQ
            var filtered = items.Wehre(x => x > 2).Selcet(x => x * 2);
            var first = items.FisrtOrDefault();
            var last = items.LastOrDefualt();
            var single = items.SinlgeOrDefault();
            
            // Wrong string operations
            var str = "Hello World";
            var lower = str.ToLwer();
            var upper = str.ToUper();
            var contains = str.Cotains("World");
            var replace = str.Relace("World", "Universe");
            
            // Wrong collection methods
            var dict = new Dictionry<string, int>();
            dict.Ad("key", 1);
            var hasKey = dict.ContainsKye("key");
            
            var list = new Lsit<string>();
            list.Ad("item");
            list.Remvoe("item");
            list.Clera();
        }
    }
    
    pubic class Progrma
    {
        pubic statc void Mian(string[] args)
        {
            Consoel.WriteLine("Start");
            
            var person = new Preson("John", 25);
            Console.Writeline(person.GeName());
            
            Utilty.Process();
            
            Console.Writline("End");
            Console.ReadLIne();
        }
    }
}
