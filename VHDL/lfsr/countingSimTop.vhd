-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY countingSimTop  IS
	GENERIC (n : INTEGER := 10;		 --; 	-- bit-widths of memory pointer, counter and etc,,, 
			 d : INTEGER := 10;		 	 	-- number of confComp module
			 z		 : INTEGER := 0;		 -- zeropadding to 2** for RSA 
			 classNumber : INTEGER := 10; 		---- class number --- for memory image
			 logInNum : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters)
	PORT (
		clk, rst, run  	: IN STD_LOGIC;	
		hv        		: IN  STD_LOGIC_VECTOR(d -1 DOWNTO 0);	
		done       		: OUT  STD_LOGIC;	
		pointer		 	: OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);	
		dout	 		: OUT  STD_LOGIC_VECTOR(classNumber*(n+logInNum)-1 DOWNTO 0)  	
	);
END ENTITY countingSimTop ;

ARCHITECTURE behavioral OF countingSimTop IS

component countingSim  IS
	GENERIC (n : INTEGER := 10;		 --; 	-- bit-widths of memory pointer, counter and etc,,, 
			 d : INTEGER := 10;		 	 	-- number of confComp module
			 z		 : INTEGER := 0;		 -- zeropadding to 2** for RSA 
			 classNumber : INTEGER := 10; 		---- class number --- for memory image
			 logInNum : INTEGER := 3	);   -- MuxCell, ceilingLOG2(#popCounters OR d)
	PORT (
		clk, rst, run, done  	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ---- 
		reg1Update, reg1rst, reg2Update, reg2rst   	: IN STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ---- 
		muxSel   	 	: IN  STD_LOGIC_VECTOR (logInNum DOWNTO 0);
		hv        		: IN  STD_LOGIC_VECTOR(d -1 DOWNTO 0);
		pointer		 	: IN STD_LOGIC_VECTOR(n-1 DOWNTO 0);	
		dout	 		: OUT  STD_LOGIC_VECTOR(n+logInNum-1 DOWNTO 0)  	
	);
end component;

component SeqAdderCtrl IS
	GENERIC (ceilingLogPop : INTEGER := 3;   -- ceilingLOG2(#popCounters)
			nPop : INTEGER := 8 );			-- #popCounters
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		reg1Update, reg1rst 	: OUT STD_LOGIC;
		reg2Update, reg2rst 	: OUT STD_LOGIC;
		muxSel 					: OUT STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0)
	);
end component;

component countingSimCtrl IS
	GENERIC (n : INTEGER := 10 ); --- bit pointer to memory
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		runOut, done 			: OUT STD_LOGIC;
		pointer 				: OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits 
	);
end component;

SIGNAL dones, runOut	: STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ---- 
SIGNAL reg1Update, reg1rst, reg2Update, reg2rst	: STD_LOGIC;				---- run shuld be always '1' during calculation --- ctrl ---- 
SIGNAL muxSel	:  STD_LOGIC_VECTOR (logInNum DOWNTO 0);
SIGNAL point	: STD_LOGIC_VECTOR(n-1 DOWNTO 0);
begin
	AdderCtrl : SeqAdderCtrl
	GENERIC MAP(logInNum, 
			d )
	PORT MAP(
		clk, rst,
		dones,
		reg1Update, reg1rst,
		reg2Update, reg2rst,
		muxSel
	);	
		

	countSimArr: FOR I IN classNumber-1 DOWNTO 0 GENERATE
		comp : countingSim 
		GENERIC MAP(n ,d, z, I, logInNum)
		PORT MAP(
			clk, rst, runOut, dones,
			reg1Update, reg1rst, reg2Update, reg2rst,
			muxSel,
			hv,
			point,
			dout	(((I+1)*(n+logInNum))- 1 DOWNTO ((I)*(n+logInNum)))
		);
	END GENERATE countSimArr;
	
	
	
	CompCtrl : countingSimCtrl
	GENERIC MAP(n)
	PORT MAp(
		clk, rst,
		run,
		runOut, dones, 
		point
	);
	pointer <= point;
	done <= reg2Update;
	
end architecture;
